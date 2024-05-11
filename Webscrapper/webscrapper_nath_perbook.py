import sys 
import requests
from bs4 import BeautifulSoup

def scrape_product_page(url):
    # Send a GET request to the URL
    response = requests.get(url)
    data_found = []
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.content, 'html.parser')
######################################################################################################
################################## FOR DEBUG PERPOSE ##################################
######################################################################################################
        #with open("raw.txt", "w") as file:
        #    file.write(soup.prettify())
                
######################################################################################################

        # Find the product title
        title_element = soup.find('h1', class_='font_2 wixui-rich-text__text')
        title = title_element.text.strip() if title_element else "Title not found"
        print("Title:", title)
        data_found.append(title)

        # Find the product old price
        price_element = soup.find('div', class_='HcOXKn c9GqVL QxJLC3 comp-lkps559k_r_comp-kq0trxmy_r_comp-lkb5m5pu2 wixui-rich-text')
        old_price = price_element.text.strip() if price_element else "Price not found"
        print("Old Price:", old_price)
        data_found.append(old_price)

        # Find the product new price
        price_element = soup.find('div', class_='comp-lkps559k_r_comp-kq0trxmy_r_comp-lkb5m5pv19')
        new_price = price_element.text.strip() if price_element else "Price not found"
        print("New Price:", new_price)
        data_found.append(new_price)


        # Find all elements with the class 'font_8' and 'wixui-rich-text__text'
        Details_elements = (soup.find_all('div', class_='font_8 wixui-rich-text__text'))

        Details_list = []

        # Iterate over each 'div' element in 'Details_elements'
        for item in Details_elements:
            # Find all 'p' elements within the current 'div' element
            p_elements = item.find_all('p')
            # Iterate over each 'p' element
            for p in p_elements:
                # Append the text of the 'p' element to the list
                Details_list.append(p.get_text(strip=True))


######################################################################################################
################################## FOR DEBUG PERPOSE ##################################
######################################################################################################
        #Print or use the text_list as needed
        #print(Details_list)
        #print(len(Details_list))

######################################################################################################
        
        [details1,details2,details3,details4] = ["-", "-", "-", "-"]
        strings_to_check = ["Hard", "Paper"]

        for element in Details_list:

            # Check if the element is a number
            if element.isdigit():
                # Check if the number is greater than or equal to 10
                if len(element) >= 10:
                    details1=element
                    print("ISBN:", details1)
                else: 
                    details2=element
                    print("Pages:", details2)

            # Check if string 
            elif isinstance(element, str):
                # Check if the string contains "Hard" or "paper"
                if any(string in element for string in strings_to_check):
                    print(f'"Binding: {element}"')
                    details3=element
                else:
                    details4=element
                    print(f'"About: {details4}"')

        data_found.append(details1)
        data_found.append(details2)
        data_found.append(details3)
        data_found.append(details4)

        return data_found
    
    else:
        print("Failed to retrieve webpage")

# URL of the product page to scrape
url = "https://www.patrabharati.com/product-page/debi-chaudhurani"



######################################################################################################
################################## FOR DEBUG PERPOSE ##################################
######################################################################################################

# Call the function to scrape the product page
#scrape_product_page(url)

######################################################################################################
