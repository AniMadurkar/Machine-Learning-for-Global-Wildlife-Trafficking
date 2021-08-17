import pandas as pd
import numpy as np
import os
import regex as re

def main():
    
    #Filepath for Panjiva data
    path = os.getcwd()
    files = os.listdir(path + '/Panjiva Data')
    panjiva_data = {}

    dtypes = {'Bill of Lading Number': str, 'Bill of Lading Type': str, 'Master Bill of Lading Number': str, 'Arrival Date': str, 
            'Matching Fields': str, 'Consignee': str, 'Consignee Address': str, 'Consignee City': str, 'Consignee State/Region': str, 
            'Consignee Postal Code': str, 'Consignee Country': str, 'Consignee Full Address': str, 'Consignee Email 1': str,  'Consignee Email 2': str, 
            'Consignee Email 3': str, 'Consignee Phone 1': str, 'Consignee Phone 2': str, 'Consignee Phone 3': str, 'Consignee Fax': str, 
            'Consignee Website 1': str, 'Consignee Website 2': str, 'Consignee Profile': str, 'Consignee D-U-N-SÂ®': float, 'Consignee Industry': str, 
            'Consignee Revenue': str, 'Consignee Employees': str, 'Consignee Market Capitalization': float, 'Consignee Incorporation Year': str, 'Consignee Trade Roles': str, 
            'Consignee SIC Codes': str, 'Consignee Stock Tickers': str, 'Consignee (Original Format)': str, 'Consignee Global HQ': str, 'Consignee Global HQ Address': str, 
            'Consignee Global HQ D-U-N-SÂ®': float, 'Consignee Domestic HQ': str, 'Consignee Domestic HQ Address': str, 'Consignee Domestic HQ D-U-N-SÂ®': float, 'Consignee Ultimate Parent': str, 
            'Consignee Ultimate Parent Website': str, 'Consignee Ultimate Parent Headquarters Address': str, 'Consignee Ultimate Parent Profile': str, 'Consignee Ultimate Parent Stock Tickers': str, 'Shipper': str,
            'Shipper Address': str, 'Shipper City': str, 'Shipper State/Region': str, 'Shipper Postal Code': str, 'Shipper Country': str,
            'Shipper Full Address': str, 'Shipper Email 1': str, 'Shipper Email 2': str, 'Shipper Email 3': str, 'Shipper Phone 1': str,
            'Shipper Phone 2': str, 'Shipper Phone 3': str, 'Shipper Fax': str, 'Shipper Website 1': str, 'Shipper Website 2': str,
            'Shipper Profile': str, 'Shipper D-U-N-SÂ®': float, 'Shipper Industry': str, 'Shipper Revenue': str, 'Shipper Employees': str,
            'Shipper Market Capitalization': float, 'Shipper Incorporation Year': str, 'Shipper Trade Roles': str, 'Shipper SIC Codes': str, 'Shipper Stock Tickers': str,
            'Shipper (Original Format)': str, 'Shipper Global HQ': str, 'Shipper Global HQ Address': str, 'Shipper Global HQ D-U-N-SÂ®': float, 'Shipper Domestic HQ': str,
            'Shipper Domestic HQ Address': str, 'Shipper Domestic HQ D-U-N-SÂ®': float, 'Shipper Ultimate Parent': str, 'Shipper Ultimate Parent Website': str, 'Shipper Ultimate Parent Headquarters Address': str,
            'Shipper Ultimate Parent Profile': str, 'Shipper Ultimate Parent Stock Tickers': str, 'Carrier': str, 'Notify Party': str, 'Notify Party SCAC': str,
            'Shipment Origin': str, 'Shipment Destination': str, 'Shipment Destination Region': str, 'Port of Unlading': str, 'Port of Unlading Region': str,
            'Port of Lading': str, 'Port of Lading Region': str, 'Port of Lading Country': str, 'Place of Receipt': str, 'Transport Method': str,
            'Vessel': str, 'Vessel Voyage ID': str, 'Vessel IMO': float, 'Is Containerized': str, 'Volume (TEU)': float,
            'Quantity': str, 'Measurement': str, 'Weight (kg)': float, 'Weight (t)': float, 'Weight (Original Format)': str,
            'Value of Goods (USD)': float, 'FROB': str, 'Manifest Number': float, 'Inbond Code': str, 'Number of Containers': float,
            'Has LCL': str, 'Container Numbers': str, 'HS Code': str, 'Goods Shipped': str, 'Volume (Container TEU)': str,
            'Container Marks': str, 'Divided/LCL': str, 'Container Type of Service': str, 'Container Types': str, 'Dangerous Goods': str}

    #Reading in Panjiva Data
    for file in files:
        print(f'Reading in panjiva file: {file}...')

        hscode = re.search(r".+?(?= -)", file).group(0)
        if hscode not in panjiva_data:
            try:
                panjiva_data[hscode] = pd.read_csv(f'Panjiva Data/{file}', dtype=dtypes, encoding='utf-8')
            except Exception as e:
                print(e)
                # print('This file needs to be looked at')
                continue
        else:
            try:
                panjiva_curr = pd.read_csv(f'Panjiva Data/{file}', dtype=dtypes, encoding='utf-8')
                #Sampling 50% of panjiva data
                # panjiva_curr = panjiva_curr.sample(frac=.5)
                panjiva_data[hscode] = panjiva_data[hscode].append(panjiva_curr, ignore_index=True, sort=False)
            except Exception as e:
                print(e)
                # print('This file needs to be looked at')
                continue
                
        panjiva_data[hscode]['hscode'] = hscode


    print(f'Cleaning panjiva file...')

    panjiva = pd.concat(panjiva_data.values(), ignore_index=True)

    #Dropping columns with all NaNs
    panjiva = panjiva.dropna(axis=1, how='all')

    #Three columns I found that are the same, but just used in different periods
    panjiva['Date'] = panjiva['Date'].fillna(panjiva['Arrival Date'])
    panjiva['Date'] = pd.to_datetime(panjiva['Date'], errors='coerce')
    panjiva['Weight (kg)'] = panjiva['Weight (kg)'].fillna(panjiva['Weight (KG)'])
    panjiva['Value (USD)'] = panjiva['Value (USD)'].fillna(panjiva['Value of Goods (USD)'])

    panjiva = panjiva.drop(columns=['Arrival Date', 'Weight (KG)', 'Value of Goods (USD)'])

    #Creating dictionary for adjusting numerical types. Currently all floats due to NaNs but leaving this allows us to change to ints if we know how to handle NaNs in the future
    num_dtypes = {'Consignee D-U-N-SÂ®': float,
                'Consignee Revenue': float,
                'Consignee Employees': float,
                'Consignee Market Capitalization': float,
                'Consignee Incorporation Year': float,
                'Consignee Global HQ D-U-N-SÂ®': float,
                'Consignee Domestic HQ D-U-N-SÂ®': float,
                'Shipper D-U-N-SÂ®': float,
                'Shipper Revenue': float,
                'Shipper Employees': float,
                'Shipper Market Capitalization': float,
                'Shipper Incorporation Year': float,
                'Shipper Global HQ D-U-N-SÂ®': float,
                'Shipper Domestic HQ D-U-N-SÂ®': float,
                'Vessel IMO': float,
                'Volume (TEU)': float,
                'Weight (kg)': float,
                'Weight (t)': float,
                'Value of Goods (USD)': float,
                'Manifest Number': float,
                'Number of Containers': float}

    def fixNumericCols(colname, new_type):
        new_col = panjiva[colname].replace(',', '', regex=True).astype(new_type)
        return new_col

    for col,dtyp in num_dtypes.items():
        try:
            panjiva[col] = fixNumericCols(col, dtyp)
        except:
            continue

    #Creating column that indicates this data source
    panjiva['panjiva'] = '1'

    #Turn the countries text into label encodings
    panjiva['lading_country_cat'] = panjiva['Port of Lading Country'].astype('category').cat.codes
    panjiva['unlading_country_cat'] = panjiva['Port of Unlading Country'].astype('category').cat.codes

    #Cleaning and renaming columns in panjiva
    panjiva.columns= panjiva.columns.str.lower()

    print(f'Writing panjiva file...')
    # panjiva_sample = panjiva.sample(frac=.01)
    # panjiva_sample.to_excel('panjiva_sample_cleaned (small).xlsx')
    panjiva.to_csv('panjiva_cleaned.csv')

if __name__ == '__main__':
    main()
