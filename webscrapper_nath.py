#WRITTEN BY: ABHISHEK NATH
#PURPOSE: bLEH

import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd  

url_list=["https://boichoi.com/product-category/bengali-books/bangladesh-book-shop/?products-per-page=all","https://boichoi.com/product-category/bengali-books/buy-academic-bengali-books-online/?products-per-page=all","https://boichoi.com/product-category/bengali-books/academic"]

url = url_list[0]

MainList_name=[]
MainList_price=[]
MainList_author=[]
MainList_price_original=[]
MainList_price_reduced=[]

page = urlopen(url)
html = page.read().decode("utf-8")
f_html = open('page_html_fullcode.txt','w')
print(html, file=f_html) 


soup = BeautifulSoup(html, features="lxml")

pattern = "Quick View\n*Futures and options"
match_results = re.findall(pattern, html, re.IGNORECASE)

f_htmlSoup = open('page_html_soupcode.txt','w') 
print(soup.get_text(),file=f_htmlSoup)


elements_NameList = soup.select('.title')

for element_name in elements_NameList:
    for child in element_name.children:
        MainList_name.append(child.contents)

elements_PriceList = soup.find_all("span",{"class": "price"})

for element_price in elements_PriceList:
    for child in element_price.select('bdi'):
            x=child
            y=x.text
            MainList_price.append(y)

elements_AuthorList = soup.select('.woo-desc')

for element_name in elements_AuthorList:
    for child in element_name.children:
        MainList_author.append(child)            

MainList_price_original=MainList_price[0::2]
MainList_price_reduced=MainList_price[1::2]

MainList=[MainList_name,MainList_author,MainList_price_original,MainList_price_reduced]

df = pd.DataFrame(MainList) 
df=df.T
df.columns =['Book', 'Author', 'Price', 'Reduced price']
writer = pd.ExcelWriter('output.xlsx')
df.to_excel(writer)
writer.save()
