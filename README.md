# Machine Learning for Global Wildlife Trafficking

This application is an extensive modeling approach for Wildlife Trafficking (live animals and wildlife products) importing via shipments into the US. You can access the application here: 
[Machine Learning for Global Wildlife Trafficking](https://share.streamlit.io/AniMadurkar/Machine-Learning-for-Global-Wildlife-Trafficking/main/streamlit_app.py)

This product leverages 2 datasets: LEMIS and Panjiva. The [LEMIS data set](https://data.nal.usda.gov/dataset/data-united-states-wildlife-and-wildlife-product-imports-2000%E2%80%932014) includes data on 15 years of the importation of wildlife and their derived products into the United States (2000–2014), originally collected by the United States Fish and Wildlife Service. The [Panjiva data set](https://panjiva.com/) was manually downloaded through a paid Panjiva account and it includes data for imported shipments (2007-2021) related to wildlife for HS codes 01, 02, 03, 04, and 05 as these represent animals & animal products.

The data used in this application is only about 1% (total, before filtering to some percentage of training data) of all data available due to Github and Streamlit size limitations.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Get a copy

Get a copy of this project by simply running the git clone command.

``` git
git clone 
```

### Prerequisites

Before running the project, we have to install all the dependencies from requirements.txt

``` pip
pip install -r requirements.txt
```
