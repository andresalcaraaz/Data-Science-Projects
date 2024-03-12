import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
import plotly.express as px

st.header("""***Company Trajectory Predictor***""")

st.sidebar.header('***Company Information***')

def user_input_features():
    def revenue_function(value):
     if value == '$10B+' :
        return 'A'
     if value == '$1B-$10B' :
        return 'B'
     if value == '$500M-$1B' :
        return 'C'
     if value == '$100M-$500M' :
        return 'D'
     if value == '$50M-$100M' :
        return 'E'
     if value == '$10M-$50M' :
        return 'F'
     if value == '$1M-$10M' :
        return 'G'
     if value == 'Less than $1M' :
        return 'H'
     else: 
        return None
            
 # Define a condition function
    def employee_category_function(value):
     if value == '1-10' :
        return 'Small'
     if value == '11-50' :
        return 'Small'
     if value == '51-100' :
        return 'Medium'
     if value == '101-250' :
        return 'Medium'
     if value == '251-500' :
        return 'Large'
     else: 
        return None
#Define a function to map countries to government types
    def government_type_func(country):
       if country == 'Australia':
          return  'Monarchy-Parliamentary Democracy'
       if country == 'Canada': 
         return 'Monarchy-Parliamentary Democracy'
       if country == 'China': 
         return 'Authoritarian Government'
       if country == 'France':
         return   'Parliamentary Democracy'
       if country == 'Germany': 
         return 'Parliamentary Democracy'
       if country == 'India': 
         return'Parliamentary Democracy'
       if country == 'Israel': 
         return 'Parliamentary Democracy'
       if country == 'Other':
         return  'Unknown'
       if country == 'Singapore': 
         return 'Parliamentary Democracy'
       if country == 'United Kingdom': 
         return 'Monarchy-Parliamentary Democracy'
       if country ==  'United States': 
         return 'Presidential Democracy',
       if country == 'Unknown':
           return 'Unknown' 
    
    founding_date = st.sidebar.slider('Founded Year', 2014, 2023, 2018)
    last_funding_date = st.sidebar.slider('Last Funding Date', 2014, 2023, 2022)
    Valuation = st.sidebar.number_input('Current Valuation',value=1000000,step=1)
    Industry = st.sidebar.selectbox('Industry',('Information Technology', 'Energy', 'Financials',
       'Consumer Discretionary', 'Communication Services', 'Other',
       'Materials', 'Health Care', 'Industrials', 'Real Estate',
       'Consumer Staples'))
    Revenue = st.sidebar.selectbox('Revenue',('$10B+', '$1B-$10B', '$500M-$1B', '$100M-$500M', '$50M-$100M', '$10M-$50M', '$1M-$10M', 'Less than $1M'))
    Employees = st.sidebar.selectbox('Number of Employees',('1-10','11-50','51-100','101-250','251-500'))
    country = st.sidebar.selectbox("Headquarters Country", ('United States','Australia','Canada','China','France','Germany','India','Israel','Singapore','United Kingdom','Other','Unknown'))
    Number_of_Founders = st.sidebar.number_input('Number of Founders', min_value= 1,step= 1)
    Number_of_Funding_Rounds = st.sidebar.number_input('Number of Funding Rounds Completed',min_value=1, step=1)
    Number_of_Platforms = st.sidebar.number_input('Number of Social Media Platforms Present In',min_value=0, step=1)
    Last_Funding_Type = st.sidebar.selectbox('Last Funding Round Received', ('Pre-Seed','Seed','Angel','Series A', 'Series B', 'Series C', 'Series D','Series E',
       'Series G', 'Series F', 'Undisclosed',
       'Venture - Series Unknown', 'Grant', 'Post-IPO Equity',
       'Convertible Note', 'Post-IPO Debt', 'Post-IPO Secondary',
       'Debt Financing', 'Corporate Round', 'Initial Coin Offering',
       'Private Equity', 'Secondary Market'))
    
    Revenue_class = revenue_function(Revenue)
    Employee_size = employee_category_function(Employees)
    government_type = government_type_func(country)
    average_time_funding_rounds = ((last_funding_date - founding_date)*365)/Number_of_Funding_Rounds
    normalized_Avergae_Time_Taken_for_funding_rounds = average_time_funding_rounds/ 999
    log_valuation = np.log(Valuation)


    data = {
    'normalized_Avergae.Time.Taken.for.funding.rounds': normalized_Avergae_Time_Taken_for_funding_rounds,
    'Log_Valuation': log_valuation,
    'Revenue.Class': Revenue_class ,
    'Employee.Size': Employee_size,
    'Industry': Industry,
    'Government Type': government_type ,
    'Founded.Year': founding_date ,
    'Number.of.Founders': Number_of_Founders,
    'Number.of.Funding.Rounds': Number_of_Funding_Rounds,
    'Number.of.Platforms': Number_of_Platforms,
    'Last.Funding.Type': Last_Funding_Type}

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

      
companies_data = pd.read_csv('Startup_Exit_Predictor_App/App_data.csv')

# Encode categorical variables
la_revenue = LabelEncoder()
la_employee_size = LabelEncoder()
la_industry = LabelEncoder()
la_government = LabelEncoder()
la_funding = LabelEncoder()
companies_data['Revenue.Class'] = la_revenue.fit_transform(companies_data['Revenue.Class'])
companies_data['Employee.Size'] = la_employee_size.fit_transform(companies_data['Employee.Size'])
companies_data['Industry'] = la_industry.fit_transform(companies_data['Industry'])
companies_data['Government Type'] = la_government.fit_transform(companies_data['Government Type'])
companies_data['Last.Funding.Type'] = la_funding.fit_transform(companies_data['Last.Funding.Type'])

df_copy = df.copy()
# Check for unseen categories
for column, encoder in [('Revenue.Class', la_revenue), ('Employee.Size', la_employee_size),
                        ('Industry', la_industry), ('Government Type', la_government),
                        ('Last.Funding.Type', la_funding)]:
    unknown_categories = set(df_copy[column]) - set(encoder.classes_)
    if unknown_categories:
        raise ValueError(f"Unseen categories in column '{column}': {unknown_categories}")


# Transform User Input data to label encoded
df['Revenue.Class'] = la_revenue.transform(df['Revenue.Class'])
df['Employee.Size'] = la_employee_size.transform(df['Employee.Size'])
df['Industry'] = la_industry.transform(df['Industry'])
df['Government Type']= la_government.transform(df['Government Type'])
df['Last.Funding.Type'] = la_funding.transform(df['Last.Funding.Type']) 
# st.subheader('Company Parameters')
# st.write(df)

# Define the features and target variable
features = ['normalized_Avergae.Time.Taken.for.funding.rounds',
            'Log_Valuation',
            'Revenue.Class',
            'Employee.Size',
            'Industry',
            'Government Type',
            'Founded.Year',
            'Number.of.Founders',
            'Number.of.Funding.Rounds',
            'Number.of.Platforms',
            'Last.Funding.Type']

target = 'Status'

# Create feature matrix (X) and target vector (y)
X = companies_data[features]
y = companies_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Use RandomUnderSampler to balance only the training set
undersampler = RandomUnderSampler(sampling_strategy='all', random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Initialize the Random Forest model
rf_model_undersample = RandomForestClassifier(n_estimators=500, max_features=4, min_samples_split=5, random_state=42)


# Train the model on the resampled training set
rf_model_undersample.fit(X_train_resampled, y_train_resampled)
y_probabilities = rf_model_undersample.predict_proba(df)  


predictions = {
    'Predicted_Active': round(y_probabilities[:, 0][0] * 100),
    'Predicted_Failed': round(y_probabilities[:, 1][0] * 100),
    'Predicted_Exit': round(y_probabilities[:, 2][0] * 100)
}

data = {
    'Private Active': predictions['Predicted_Active'],
    'Closed': predictions['Predicted_Failed'],
    'Exit (M&A or IPO)': predictions['Predicted_Exit']
}

# Extract labels and percentages
labels = list(data.keys())[0:]
percentages = list(data.values())[0:]

# Convert data to a DataFrame
df = pd.DataFrame([data])

# Set McKinsey color palette
colors = ['#30A3DA','#0000ff','#051C2A']
# Specify the order of categories for color assignment
category_order = [labels[0], labels[1], labels[2]]

# Create the donut chart using Plotly Express
fig = px.pie(
    df,
    names=labels,
    values=percentages,
    color_discrete_sequence=colors,
    hole=0.5,  # Set the size of the hole for the donut chart
    category_orders={'names': category_order}
)
fig.update_layout(width=800, height=600)
fig.update_layout(legend=dict(font=dict(size=20)))
fig.update_traces(textinfo='percent', insidetextfont=dict(size=20))


# Display the plot
st.plotly_chart(fig)





