# Used to check groceries' flyers

import getFlyer_sobeys_walmart,data_clean

def sobeys_walmart_flyer(url):
    draft_flyer_file,shopName = getFlyer_sobeys_walmart.getFlyer(url)
    if draft_flyer_file == -1:
        print("error in getting flyer data")
        return
    data_clean.clean_data(draft_flyer_file,shopName)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    urls = ["https://www.sobeys.com/en/flyer/","https://www.walmart.ca/en/flyer"]
    for url in urls:
        sobeys_walmart_flyer(url)

