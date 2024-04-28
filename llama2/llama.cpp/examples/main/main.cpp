// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "console.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <chrono>
#include <thread>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting = false;

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>


// Function We write for read content from "input.txt" as the prompt for LLama Model
std::string ReadAndClearFile(const std::string &filename) {
    // Open the file and read its contents
    std::ifstream inFile(filename);
    std::string content((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    //std::cout << "ths is content: " << content << "\n"<< std::endl;
    inFile.close();

    // Clear the file
    std::ofstream outFile(filename);
    outFile.close();

    return content;
}

void saveConversation(const std::string& input) {
                    std::ofstream outputFile;
                    outputFile.open("./../../output/conversation/conversation_record.csv", std::ios::app);
                    if (outputFile.is_open()) {
                        outputFile << input << "\n";
                        outputFile.close();
                    } else {
                        std::cout << "Failed to open the CSV file for writing." << std::endl;
                    }
                }

// For write the output token into the output.txt as the response for robot
// Function to write content to a text file
void WriteToFile(const std::string& filename,  const std::string output) {
    std::ofstream file(filename, std::ios::out); // Open the file in write mode. If the file does not exist, it will be created.

    // Check if the file is open
    if (file.is_open()) {
        // Write the content to the file
        //file << "This is a test";
        // In case the token will return "" but it is tokenized the Buffer
        if (output != "") {
        file << output;  // .c_str() is not needed for std::string
        }
        // Close the file
        file.close();
    } else {
        // Handle the case where the file could not be opened
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}


// Write to cutomized the output response which will send to Pepper
std::string extractSentence(const std::string& input,const std::string& filename) {
    // Find the starting position of "Bob:"
    size_t start = input.find("Bob: ");
    // Find the starting position of "\n User:"
    size_t end = input.find("User:");

//    // Ensure both substrings are found and "Bob:" comes before "\n User:"
//    if (start != std::string::npos && end != std::string::npos && start < end) {
//        // Adjust start position to the end of "Bob: "
//        start += 5; // Length of "Bob: " is 5
//        // Extract the content between
//        return input.substr(start, end - start);
//    }
    if (start != std::string::npos){
        if (end != std::string::npos){
            std::string result = input.substr(start + 5,end);
             std::cout << "ths is response: " << result << "\n"<< std::endl;
             WriteToFile(filename,result);
             return result;
        }
        else{
        std::string result = input.substr(start + 5);
         WriteToFile(filename,result);
         return result;
         }

    }

    std::cout << "Bob not find in the string\n"<< std::endl;

    // Return an empty string if the pattern is not found
    return "empty result";
}

void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const std::vector<llama_token> input_tokens, const std::string output, const std::vector<llama_token> output_tokens) {

    if (params.logdir.empty()) {
        return;
    }

    const std::string timestamp = get_sortable_timestamp();

    const bool success = create_directory_with_parents(params.logdir);
    if (!success) {
        fprintf(stderr, "%s: warning: failed to create logdir %s, cannot write logfile\n",
                __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: main\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    dump_non_result_info_yaml(logfile, params, ctx, timestamp, input_tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Generation Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    dump_string_yaml_multiline(logfile, "output", output.c_str());
    dump_vector_int_yaml(logfile, "output_tokens", output_tokens);

    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile);
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting = true;
        } else {
            console::cleanup();
            printf("\n");
            llama_print_timings(*g_ctx);
            write_logfile(*g_ctx, *g_params, *g_model, *g_input_tokens, g_output_ss->str(), *g_output_tokens);
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    gpt_params params;
    g_params = &params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("main", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc,argv);
#endif // LOG_DISABLE_LOGS

    // TODO: Dump params ?
    //LOG("Params perplexity: %s\n", LOG_TOSTR(params.perplexity));

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 10000.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx > 2048) {
        // TODO: determine the actual max context of the model (e.g. 4096 for LLaMA v2) and use that instead of 2048
        LOG_TEE("%s: warning: base model only supports context sizes no greater than 2048 tokens (%d specified)\n", __func__, params.n_ctx);
    } else if (params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    LOG_TEE("%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;
    llama_context * ctx_guidance = NULL;
    g_model = &model;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (params.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_ctx parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            LOG_TEE("%s: testing memory usage for n_batch = %d, n_ctx = %d\n", __func__, params.n_batch, params.n_ctx);

            const std::vector<llama_token> tmp(params.n_batch, llama_token_bos(ctx));
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_ctx, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);
        llama_free_model(model);

        return 0;
    }

    // export the cgraph and exit
    if (params.export_cgraph) {
        llama_eval_export(ctx, "llama.ggml");
        llama_free(ctx);
        llama_free_model(model);

        return 0;
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_TEE("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_TEE("%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            llama_set_rng_seed(ctx, params.seed);

            LOG_TEE("%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            LOG_TEE("%s: session file does not exist, will create\n", __func__);
        }
    }

    const bool add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM;
    LOG("add_bos: %d\n", add_bos);

    std::vector<llama_token> embd_inp;

    if (params.interactive_first || params.instruct || !params.prompt.empty() || session_tokens.empty()) {
        LOG("tokenize the prompt\n");
        embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos);
    } else {
        LOG("use session tokens\n");
        embd_inp = session_tokens;
    }

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp));

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(ctx));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp));
    }

    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    if (ctx_guidance) {
        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(params.cfg_negative_prompt));

        guidance_inp = ::llama_tokenize(ctx_guidance, params.cfg_negative_prompt, add_bos);
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp));

        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, add_bos);
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp));

        original_prompt_len = original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
        LOG("original_prompt_len: %s", log_tostr(original_prompt_len));
        LOG("guidance_offset:     %s", log_tostr(guidance_offset));
    }

    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (session_tokens.size() > 0) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_TEE("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_TEE("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_TEE("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_TEE("%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    LOGLN(
            "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu",
            log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOGLN("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode
    const auto inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", add_bos);
    const auto inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n",    false);

    LOG("inp_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_pfx));
    LOG("inp_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_sfx));

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive_first = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_TEE("\n");
        LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (ctx_guidance) {
            LOG_TEE("\n");
            LOG_TEE("%s: negative prompt: '%s'\n", __func__, params.cfg_negative_prompt.c_str());
            LOG_TEE("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (int i = 0; i < (int) guidance_inp.size(); i++) {
                LOG_TEE("%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
            }
        }

        if (params.n_keep > 0) {
        LOG_TEE("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_TEE("'\n");
        }
        LOG_TEE("\n");
    }

    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

        LOG_TEE("%s: interactive mode on.\n", __func__);

        if (params.antiprompt.size()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_TEE("Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (params.input_prefix_bos) {
            LOG_TEE("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_TEE("Input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (!params.input_suffix.empty()) {
            LOG_TEE("Input suffix: '%s'\n", params.input_suffix.c_str());
        }
    }
    LOG_TEE("sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
            params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_TEE("\n\n");

    grammar_parser::parse_state parsed_grammar;
    llama_grammar *             grammar = NULL;
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return 1;
        }
        LOG_TEE("%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        LOG_TEE("\n");

        {
            auto it = params.logit_bias.find(llama_token_eos(ctx));
            if (it != params.logit_bias.end() && it->second == -INFINITY) {
                LOG_TEE("%s: warning: EOS token is disabled, which will cause most grammars to fail\n", __func__);
            }
        }

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        grammar = llama_grammar_init(
            grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        const char *control_message;
        if (params.multiline_input) {
            control_message = " - To return control to LLaMa, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to LLaMa.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_TEE("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_TEE(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_TEE(       "%s\n", control_message);

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;


    // Add one varible to store the outputToken
    std::string responseTokens;

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;

    {
        LOG("warming up the model with an empty run\n");

        const std::vector<llama_token> tmp = { llama_token_bos(ctx), };
        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        llama_reset_timings(ctx);
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (embd.size() > 0) {
            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(console::error);
                printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(console::reset);
                fflush(stdout);
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() + std::max<int>(0, guidance_offset) > n_ctx) {
                if (params.n_predict == -2) {
                    LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                    break;
                }

                const int n_left = n_past - params.n_keep;
                LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d\n", n_past, n_left, n_ctx, params.n_keep);

                // always keep the first token - BOS
                n_past          = std::max(1, params.n_keep);
                n_past_guidance = std::max(1, params.n_keep + guidance_offset);

                LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd));

                LOG("clear session path\n");
                path_session.clear();
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always

            if (ctx_guidance) {
                int input_size = 0;
                llama_token * input_buf = NULL;

                if (n_past_guidance < (int) guidance_inp.size()) {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd.begin() + original_prompt_len < embd.end()) {
                        embd_guidance.insert(
                            embd_guidance.end(),
                            embd.begin() + original_prompt_len,
                            embd.end()
                        );
                    }

                    input_buf  = embd_guidance.data();
                    input_size = embd_guidance.size();

                    LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_guidance));
                } else {
                    input_buf  = embd.data();
                    input_size = embd.size();
                }

                for (int i = 0; i < input_size; i += params.n_batch) {
                    int n_eval = std::min(input_size - i, params.n_batch);
                    if (llama_eval(ctx_guidance, input_buf + i, n_eval, n_past_guidance, params.n_threads)) {
                        LOG_TEE("%s : failed to eval\n", __func__);
                        return 1;
                    }

                    n_past_guidance += n_eval;
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd));

                if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG("n_past = %d\n", n_past);
            }

            if (embd.size() > 0 && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
            }
        }

        embd.clear();
        embd_guidance.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            const float   temp            = params.temp;
            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
            const float   top_p           = params.top_p;
            const float   tfs_z           = params.tfs_z;
            const float   typical_p       = params.typical_p;
            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
            const float   repeat_penalty  = params.repeat_penalty;
            const float   alpha_presence  = params.presence_penalty;
            const float   alpha_frequency = params.frequency_penalty;
            const int     mirostat        = params.mirostat;
            const float   mirostat_tau    = params.mirostat_tau;
            const float   mirostat_eta    = params.mirostat_eta;
            const bool    penalize_nl     = params.penalize_nl;

            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG("saved session to %s\n", path_session.c_str());
            }

            llama_token id = 0;

            {
                auto logits  = llama_get_logits(ctx);
                auto n_vocab = llama_n_vocab(ctx);

                // Apply params.logit_bias map
                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                    logits[it->first] += it->second;
                }

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);
                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array cur_p = { candidates.data(), candidates.size(), false };

                if (ctx_guidance) {
                    llama_sample_classifier_free_guidance(ctx, &cur_p, ctx_guidance, params.cfg_scale);
                }

                // Apply penalties
                float nl_logit = logits[llama_token_nl(ctx)];
                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                llama_sample_repetition_penalty(ctx, &cur_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, repeat_penalty);
                llama_sample_frequency_and_presence_penalties(ctx, &cur_p,
                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                    last_n_repeat, alpha_frequency, alpha_presence);
                if (!penalize_nl) {
                    for (size_t idx = 0; idx < cur_p.size; idx++) {
                        if (cur_p.data[idx].id == llama_token_nl(ctx)) {
                            cur_p.data[idx].logit = nl_logit;
                            break;
                        }
                    }
                }

                if (grammar != NULL) {
                    llama_sample_grammar(ctx, &cur_p, grammar);
                }

                if (temp <= 0) {
                    // Greedy sampling
                    id = llama_sample_token_greedy(ctx, &cur_p);
                } else {
                    if (mirostat == 1) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        const int mirostat_m = 100;
                        llama_sample_temperature(ctx, &cur_p, temp);
                        id = llama_sample_token_mirostat(ctx, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                    } else if (mirostat == 2) {
                        static float mirostat_mu = 2.0f * mirostat_tau;
                        llama_sample_temperature(ctx, &cur_p, temp);
                        id = llama_sample_token_mirostat_v2(ctx, &cur_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                    } else {
                        // Temperature sampling
                        llama_sample_top_k      (ctx, &cur_p, top_k, 1);
                        llama_sample_tail_free  (ctx, &cur_p, tfs_z, 1);
                        llama_sample_typical    (ctx, &cur_p, typical_p, 1);
                        llama_sample_top_p      (ctx, &cur_p, top_p, 1);
                        llama_sample_temperature(ctx, &cur_p, temp);

                        {
                            const int n_top = 10;
                            LOG("top %d candidates:\n", n_top);

                            for (int i = 0; i < n_top; i++) {
                                const llama_token id = cur_p.data[i].id;
                                LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx, id).c_str(), cur_p.data[i].p);
                            }
                        }

                        id = llama_sample_token(ctx, &cur_p);

                        // Store the Response Token
                        responseTokens += llama_token_to_piece(ctx, id).c_str();

                        //LOG("responseTokens:'%s'\n", responseTokens.c_str());
                        LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx, id).c_str());
                    }
                }
                // printf("`%d`", candidates_p.size);

                // std::string outFilename = "in_output/output.txt";
//                std::string outFilename = "./../../output/exchange_information/cpp_to_py.txt";
//                std::string response = extractSentence(responseTokens);
//                WriteToFile(outFilename,responseTokens);
                //std::cout<< "This is the original token  " << responseTokens <<" \n"<< std::endl;
                //std::cout<< "This is the adjusted response from llama   " << response <<" \n"<< std::endl;
                //WriteToFile(outFilename,responseTokens);
                //responseTokens = "";
//                std::ofstream outPipe("./../../client/pipe_cpp_to_py"); // Stores in Root Directory
//                if (outPipe.is_open()) {
//                    outPipe << response << std::endl;
//                    //std::cout << response<< std::endl;
//                    outPipe.close();
//                    }
                //saveConversation(response); // Write response into the csv file to record


                if (grammar != NULL) {
                    llama_grammar_accept_token(ctx, grammar, id);
                }

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, last_n_tokens));
            }

            embd.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo) {
            for (auto id : embd) {
                const std::string token_str = llama_token_to_piece(ctx, id);
                printf("%s", token_str.c_str());

                if (embd.size() > 1) {
                    input_tokens.push_back(id);
                } else {
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(console::reset);
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt
            if (params.antiprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_piece(ctx, id);
                }

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                            console::set_display(console::user_input);
                        }
                        is_antiprompt = true;
                        fflush(stdout);
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of text token in interactive mode
            if (last_n_tokens.back() == llama_token_eos(ctx)) {
                LOG("found EOS token\n");

                if (params.interactive) {
                    if (params.antiprompt.size() != 0) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    is_interacting = true;
                    printf("\n");
                    console::set_display(console::user_input);
                    fflush(stdout);
                } else if (params.instruct) {
                    is_interacting = true;
                }
            }

            if (n_past > 0 && is_interacting) {
                LOG("waiting for user input\n");

                if (params.instruct) {
                    printf("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(ctx));
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }


                //std::string line;
                //bool another_line = true;
                //do {
                  //  another_line = console::readline(line, params.multiline_input);
                    //buffer += line;
                //} while (another_line);

                std::string outFilename = "./../../output/exchange_information/cpp_to_py.txt";
                std::string response = extractSentence(responseTokens,outFilename);
//                if (response != 'empty result'){
//                    WriteToFile(outFilename,response);}
                //std::cout<< "This is the original token  " << responseTokens <<" \n"<< std::endl;
                responseTokens = "";

                //std::string filename = "in_output/input.txt";
                 std::string filename = "./../../output/exchange_information/py_to_cpp.txt";
                 bool another_line = true;
                 while (another_line) {
                     std::this_thread::sleep_for(std::chrono::seconds(1));  // wait for 1 second
                     std::string content = ReadAndClearFile(filename);
                     if (!content.empty()) {
                         another_line = false;
                         buffer += content;
                     }
                 }
//                std::string content;
//                content = "";
//                std::string py_to_cpp_path = "./../../client/test";
//                std::cout << "Start receiving infor" << std::endl;
//               while(content.length() == 0){
//
//               std::ifstream inPipe(py_to_cpp_path); // Receive the transcripr from whisper
//                   getline(inPipe,content);
//                   buffer += content;
//                   std::cout << "Receove from py: "<< content << std::endl;
//                   //inPipe.clear();
//                   inPipe.close();
//               }

                // Inside the code block where you want to save the conversation
                //saveConversation(content);


                // done taking input, reset color
                console::set_display(console::reset);

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        buffer += params.input_suffix;
                        printf("%s", params.input_suffix.c_str());
                    }

                    LOG("buffer: '%s'\n", buffer.c_str());
                    std::cout << "Buffer is " << buffer <<std::endl;
                    const size_t original_size = embd_inp.size();

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        LOG("inserting instruction prefix\n");
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    const auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp));

                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        LOG("inserting instruction suffix\n");
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << llama_token_to_piece(ctx, token);
                    }

                    n_remain -= line_inp.size();
                    LOG("n_remain: %d\n", n_remain);
                } else {
                    LOG("empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    // reset grammar state if we're restarting generation
                    if (grammar != NULL) {
                        llama_grammar_free(grammar);

                        std::vector<const llama_grammar_element *> grammar_rules( parsed_grammar.c_rules());
                        grammar = llama_grammar_init(
                            grammar_rules.data(), grammar_rules.size(),
                            parsed_grammar.symbol_ids.at("root"));
                    }
                }
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(ctx) && !(params.instruct || params.interactive)) {
            LOG_TEE(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }




    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG_TEE("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    llama_print_timings(ctx);
    write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);



    if (ctx_guidance) { llama_free(ctx_guidance); }
    llama_free(ctx);
    llama_free_model(model);

    if (grammar != NULL) {
        llama_grammar_free(grammar);
    }
    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n")
#endif // LOG_DISABLE_LOGS

    return 0;
}
