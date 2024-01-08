import requests

# If this doesn't work, run the following in Linux command line:
# sudo kill $(sudo lsof -t -i:5000)
# or
# kill $(lsof -t -i:5000)
def quick_shutdown():
    headers = {'content-type': "/setup/end"}
    response = requests.post("http://localhost:5000" + headers["content-type"], headers=headers)
    print(response.text)
quick_shutdown()