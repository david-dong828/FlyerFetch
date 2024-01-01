# Used to check groceries' flyers

import getSobeysFlyer,data_clean

def sobeys_flyer(url):
    draft_flyer_file = getSobeysFlyer.getFlyer(url)
    data_clean.clean_sobeys_data(draft_flyer_file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sobeys_flyer_url = "https://www.sobeys.com/en/flyer/"
    sobeys_flyer(sobeys_flyer_url)

