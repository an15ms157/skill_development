# WEB SCRAPPING: Use this code as a base to start collecting data from website(s)

# HOW TO: Paste the link of webpage in the variable BASE_URL.
#         Set PAGENUMBER_START and PAGENUMBER_END as required
#         Make sure your variable has the ending "page=" 

# To run the file, use COMMAND: python3 webscrapper_nath_perpage.py 
# Check if you have all the necessary packages installed. 
# OUTPUT is produced in file : data.xlsx 
# WRITTEN BY: Abhishek Nath



import os 
import requests
from bs4 import BeautifulSoup
from webscrapper_nath_perbook import scrape_product_page
import pandas as pd



######################################################################################################
################################## PARAMETERS TO CONTROL ##################################
######################################################################################################

PAGENUMBER_START=5
PAGENUMBER_END=10
BASE_URL="https://www.patrabharati.com/category/all-products?page="

######################################################################################################





######################################################################################################
################################## FIXED VARIABLES ##################################
######################################################################################################

# Define column names and data
columns = ["URL", "Title", "Old Price", "New Price", "ISBN", "Pages", "Binding", "About"]
all_data=[]

######################################################################################################


for i in range (PAGENUMBER_START, PAGENUMBER_END):
    print("This is page number ", i)
    # URL of the webpage
    url = BASE_URL+str(i)

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all elements with class 'Jzh04F'
        elements = soup.find_all(class_='JPDEZd')
        print("Number of content in this page = ",len(elements))
        # Extract and print the href attributes for elements with class 'Jzh04F'
        for element in elements:
            href = element.get('href')
            if href:
                print(href)
                data_obtained = scrape_product_page(href)
                data_obtained.insert(0, href)
                print (data_obtained)
                all_data.append(data_obtained)
    else:
        print("Failed to fetch the page")

# Create a DataFrame
df = pd.DataFrame(all_data, columns=columns)

# Write the DataFrame to an Excel file
df.to_excel("data.xlsx", index=False)