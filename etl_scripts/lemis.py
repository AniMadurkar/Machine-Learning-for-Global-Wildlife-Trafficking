import pandas as pd
import numpy as np

def main():

    dtypes = {'control_number': int, 'species': str, 'code': str, 'taxa': str, 'class': str, 
            'genus': str, 'species': str, 'subspecies': str, 'specific_name': str, 'generic_name': str,
            'description': str, 'quantity': float, 'unit': str, 'value': float, 'country_origin': str,
            'country_imp_exp': str, 'purpose': str, 'source': str, 'action': str, 'disposition': str,
            'disposition_date': str, 'disposition_year': str, 'shipment_date': str, 'shipment_year': str, 'import_export': str,
            'port': str, 'us_co': str, 'foreign_co': str, 'cleaning_notes': str}

    #Reading Lemis files in
    print(f'Reading in lemis file...')
    lemis = pd.read_csv('Lemis Data/lemis_2000_2014_cleaned.csv', dtype=dtypes)
    codes = pd.read_csv('Lemis Data/lemis_codes.csv')

    print(f'Cleaning lemis file...')

    #Dropping unneeded columns and replacing nan value
    lemis = lemis.drop(columns = ['cleaning_notes', 'import_export'])
    lemis.at[99640, 'action'] = 'C'
    #Replace NA dates fields with value
    lemis['disposition_date'] = lemis['disposition_date'].replace('NA', '1/1/2999')
    lemis['shipment_date'] = lemis['shipment_date'].replace('NA', '1/1/2999')
    #Replace NA text values with blanks
    lemis = lemis.replace('NA', '')

    #Function to create a dataframe for each column in codes to join on
    #Skipping country for now, may want to revisit and add that in
    def createCodeDataframe(col, new_name):
        df = codes[codes['field'] == col].rename(columns={'value': new_name})
        return df

    action = createCodeDataframe('action', 'action_value')
    description = createCodeDataframe('description', 'description_value')
    disposition = createCodeDataframe('disposition', 'disposition_value')
    port = createCodeDataframe('port', 'port_value')
    purpose = createCodeDataframe('purpose', 'purpose_value')
    source = createCodeDataframe('source', 'source_value')
    unit = createCodeDataframe('unit', 'unit_value')

    #Looping through the code dataframes to join onto the main lemis df. Dropping the duplicate fields that come in as well
    for code_df in [action, description, disposition, port, purpose, source, unit]:
        lemis = lemis.merge(code_df, how='left', left_on=code_df.field.iloc[0], right_on='code')
        try:
            lemis = lemis.drop(columns=['field_x', 'code_x', 'post_feb_2013_x', 'field_y', 'code_y', 'post_feb_2013_y'])
        except:
            continue

    #Creating column that indicates this data source
    lemis['panjiva'] = '0'

    #Expanding unit value into columns and rejoining to main dataframe
    lemis.unit_value = lemis.unit_value.fillna('unknown')
    lemis_unit = lemis.set_index([lemis.index, 'unit_value'])['quantity'].unstack()
    lemis = lemis.join(lemis_unit, how='left')

    #dropping unneeded code columns and renaming
    lemis = lemis.drop(columns=['action', 'description', 'disposition', 'port', 'purpose', 'source', 'unit'])
    lemis = lemis.rename(columns={'shipment_date': 'date', 'us_co': 'consignee',
                                'action_value': 'action', 'description_value': 'description', 'disposition_value': 'disposition',
                                'port_value': 'port', 'purpose_value': 'purpose', 'source_value': 'source', 'unit_value': 'unit',
                                'Kilograms': 'weight (kg)'})

    #Cleaning and renaming columns in lemis
    lemis.columns= lemis.columns.str.lower()

    #Function to remove punctuation from us_co and foreign_co columns
    def remove_punctuation(x):
        #Lower case
        x = x.str.lower()
        #Remove all punctuation
        x = x.str.replace('[^\w\s]','', regex=True)
        #Remove more than one space into just one space
        x = x.str.replace('\s+', ' ', regex=True)
        return x
    
    #Apply function to both us_co and foreign_co columns
    lemis['consignee'] = remove_punctuation(lemis['consignee'])
    lemis['foreign_co'] = remove_punctuation(lemis['foreign_co'])

    #Creating volume column
    lemis['volume (teu)'] = lemis['cubic meters']/38.5

    print(f'Writing lemis file...')
    #Getting 1% of the data to write to file. Comment out next two lines and uncomment last line if not needed
    lemis_sample = lemis.sample(frac=.01)
    lemis_sample.to_csv('lemis_cleaned_sample.csv')
#     lemis.to_csv('lemis_cleaned.csv')

if __name__ == '__main__':
    main()
