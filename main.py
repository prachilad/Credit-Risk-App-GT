from pickletools import int4
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import classification_report
from joblib import load



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scikitplot as skplt
import pickle
# import pyodbc
import time
import os

from lime import lime_tabular

import os

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_directory = os.path.dirname(script_path)

absoulute_path = os.path.abspath(script_directory)

db_file_path = script_directory + '\Graciepoint HNWI Loan Storage1.accdb'

## Load Data
model_df = pd.read_csv(r"Model_Data.csv")

X_train, X_test, Y_train, Y_test = train_test_split(model_df.iloc[:, :6], model_df[['Customer_Status_60']], train_size=0.8, random_state=123)

## Load Model
lr_classif = pickle.load(open('trained_logit_model.sav', 'rb'))

Y_test_preds = lr_classif.predict(X_test)

## Dashboard
st.title("Credit Worthiness Prediction - Model Dashboard and Input Form ")
original_title = '<p style="font-size: 20px;">Latest Data Refresh as of 12/04/2023</p>'
st.markdown(original_title, unsafe_allow_html=True)
## st.markdown(<font-size: 20px> "Latest Refresh as of 12/04/2023")
original_title = '<p style="font-size: 30px;">Predicting PD using Credit Profile</p>'

tab1, tab2, tab3, tab4 = st.tabs(["Credit Underwriting Input Form", "Loan Scenario Testing", "Model Performance & Data", "Risk Appetite & Spread"])


# with tab1:
#     def get_connection():
#         conn_string = (f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}}; DBQ={db_file_path}')
#         try:
#                conn = pyodbc.connect(conn_string)
#                return conn
#         except Exception as e:
#                st.error(f"Error: {e}")
#                return None

#     conn = get_connection()
#     if conn:
#       spread = pd.read_sql("SELECT * FROM Spread", conn)
#       carrier = pd.read_sql("SELECT * FROM Carrier", conn)
#       conn.commit()
#       conn.close()

loaded_model = pickle.load(open('trained_logit_model.sav', 'rb'))
def credit_status_prediction(input_data, kyc_or_aml, verified_liquidity_coverage, liquidity):
    kyc_or_aml_verified_as_clear = str(kyc_or_aml)
    verified_liquidity_coverage_3_years = float(verified_liquidity_coverage)
    total_liquidity = int(float(liquidity))
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    probability = loaded_model.predict_proba(input_data_reshaped)[0, 1]

    # probability = round(probs[prediction[0]], 5)
    
    
    if (kyc_or_aml_verified_as_clear == "Yes") and (verified_liquidity_coverage_3_years >= 1.5) and (total_liquidity >= 5000000):
        if (prediction[0]==0):
            if (probability) <= (spread.iat[4,2] / 100) and (probability) > (spread.iat[3,2] / 100):
                return 'Application has cleared acceptance criteria. Probability of default for applicant of ' + str(round(probability*100, 2)) + '% ' + 'and Spread Applied to Final Rate of ' + str(spread.iat[4,3]) + '%'
                if (probability) <= (spread.iat[3,2] / 100) and (probability) > (spread.iat[2,2] / 100):
                    return 'Application has cleared acceptance criteria. Probability of default for applicant of ' + str(round(probability*100, 2)) + '% ' + 'and Spread Applied to Final Rate of ' + str(spread.iat[3,3]) + '%'
                if (probability) <= (spread.iat[2,2] / 100) and (probability) > (spread.iat[1,2] / 100):
                    return 'Application has cleared acceptance criteria. Probability of default for applicant of ' + str(round(probability*100, 2)) + '% ' + 'and Spread Applied to Final Rate of ' + str(spread.iat[2,3]) + '%'
                if (probability) <= (spread.iat[1,2] / 100) and (probability) > (spread.iat[0,2] / 100):
                    return 'Application has cleared acceptance criteria. Probability of default for applicant of ' + str(round(probability*100, 2)) + '% ' + 'and Spread Applied to Final Rate of ' + str(spread.iat[1,3]) + '%'
                if (probability) <= (spread.iat[0,2] / 100):
                    return 'Application has cleared acceptance criteria. Probability of default for applicant of ' + str(round(probability*100, 2)) + '% ' + 'and Spread Applied to Final Rate of ' + str(spread.iat[0,3]) + '%'
                else:
                    return 'Loan should not be approved, individual is a default risk with probability of default of ' + str(round(probability*100, 2))+ '% '
            else:
                return 'Loan should not be approved, individual is a default risk with probability of default of ' + str(round(probability*100, 2))+ '% '
    elif (kyc_or_aml_verified_as_clear == "No"):
        return 'Loan can not be processed due to KYC/AML status'
    elif (total_liquidity < 5000000):
        return 'Loan can not be processed due to insufficient total liquidity'
    elif (verified_liquidity_coverage_3_years < 1.5):
        return 'Loan can not be processed due to insufficient liquidity coverage'
    else:
        return 'Loan can not be processed due to one or more criteria not being met'
        


    

    

# Main streamlit app
def main():
    st.markdown("This tab contains the input sheet to calculate the probability of default for a new credit applicant. Please fill out the input sheet below to calculate the probability of default of the individual")
    st.header("Probability of Default Input Form")
    col1, col2 = st.columns(2)
    with col1:
        Average_Age_of_Secured_Trades = st.text_input(":blue[Average Age of Secured Trades in Months* (e.g. 28)]", help = "Average Age of all credit transactions with securing collateral")
        if Average_Age_of_Secured_Trades == "":
            Average_Age_of_Secured_Trades = 0
        Number_of_30_DPD_Delinquency_Occurrences_on_Trades_in_24_Months = st.text_input(label = ':blue[Number of 30 days Past Due Delinquencies* (e.g. 1)]', help = "Number of credit trades that are currently 30 days past due  delinquency (DPD) reported in 24 months")
        if Number_of_30_DPD_Delinquency_Occurrences_on_Trades_in_24_Months == "":
            Number_of_30_DPD_Delinquency_Occurrences_on_Trades_in_24_Months = 0
        Number_of_Open_Trades = st.text_input(':blue[Number of Open Loans* (e.g. 11)]', help = "Number of trades Oopened in 24 months")
        if Number_of_Open_Trades == "":
            Number_of_Open_Trades = 0
        Number_of_Revolving_Trades_with_Unpaid_Major_Derogatory_Reported_in_24_Months = st.text_input(':blue[Number of Revolving Accounts with Derogatory Reports*  (e.g. 0)]', help = "Number of Revolving Trades with Unpaid Major Derogatory Reports. Derogatory is the term used to describe negative information that is more than 180 days late. Accounts that are less than 180 days late are referred to as delinquent. Examples of derogatory accounts include collections, charge-offs, foreclosures and repossessions.")
        if Number_of_Revolving_Trades_with_Unpaid_Major_Derogatory_Reported_in_24_Months == "":
            Number_of_Revolving_Trades_with_Unpaid_Major_Derogatory_Reported_in_24_Months = 0
        Total_Balance_on_Trades_Reported_in_6_Months_pre = st.text_input(':blue[Total Balance on Trades Reported in 6 Months* (e.g. $1,100,000)]', help = "Total Balance of new credit activity reported in last 6 months")
        Total_Balance_on_Trades_Reported_in_6_Months = Total_Balance_on_Trades_Reported_in_6_Months_pre.replace(",", "").replace("$", "")
        if Total_Balance_on_Trades_Reported_in_6_Months == "":
            Total_Balance_on_Trades_Reported_in_6_Months = 0
        Monthly_Disposable_Income_w_Current_Debt = st.text_input(':blue[Debt to Income Ratio*  (e.g. 2.1)]', help = "Monthly disposable income compared to current debt")
        if Monthly_Disposable_Income_w_Current_Debt == "":
            Monthly_Disposable_Income_w_Current_Debt = 0
        kyc_or_aml_verified_as_clear = st.text_input('KYC / AML - Verified as Clear*  (e.g. Yes/No)', help = "Know you client  or  anti money laundering review flag.    Yes = Clear of any issues N=One or more issues were found in review.")
        total_liquidity_pre = st.text_input('Total Liquidity* (e.g. $11,000,000)', help = "Total verifiable liquidity from underwriting review.")
        total_liquidity = total_liquidity_pre.replace(",", "").replace("$", "")
        if total_liquidity == "":
            total_liquidity = 0
        
        
    
    credit_status = ''

    with col2:
        verified_liquidity_coverage_3_years_pre = st.text_input('Verified Liquidity Coverage - 3 Years* (e.g. 3.0)', help = "Total liquidity coverage. Calculated as verified liquid assets accessible to borrower to meet interest coverage for GP loan as well as nay collateral Gaps over the next 3 years. This measure helps to ensure their ongoing ability to meet GPs short-term obligations")
        verified_liquidity_coverage_3_years = verified_liquidity_coverage_3_years_pre.replace("%", "").replace(",", "")
        if verified_liquidity_coverage_3_years == "":
            verified_liquidity_coverage_3_years = int(0)
        #insuarance_carrier = st.selectbox('Insurance Carrier* (e.g. A, Refer to List)', (list(carrier['Carrier'])), help = "Insurance Carrier identified as carrier on record for underwritten policy.")
        loan_amount = st.text_input('Loan Amount* (e.g. $210,000)', help = "Total Loan amount")
        stated_net_worth_pre = st.text_input('Stated Net Worth (e.g. $11,000,000)', help = "Net worth started by borrower")
        stated_net_worth = stated_net_worth_pre.replace("$", "").replace(",", "")
        verified_net_worth_pre = st.text_input('Verified Net Worth* (e.g. $10,000,000)', help = "Net worth of borrower that has been verified")
        verified_net_worth = verified_net_worth_pre.replace("$", "").replace(",", "")
        verified_liabilities_pre = st.text_input('Verified Liabilities (e.g. $100,000)', help = "Amount of liabilities that have been verified by for the borrower")
        verified_liabilities = verified_liabilities_pre.replace("$", "").replace(",", "")
        most_recent_agi_pre = st.text_input('Most Recent AGI (e.g. $2,100,000)', help = "Adjusted Gross Income (most recent) for borrower")
        most_recent_agi = most_recent_agi_pre.replace("$", "").replace(",", "")
        credit_score_of_sponsors_1 = st.text_input('Credit Score of Applicant* (e.g. 730)', help = "Advantage score for borrower from Equifax")
        #sofr_rate_pre = st.text_input('SOFR Rate* (e.g. 5.5%)', help = "Current SOFR Rate (Secured Overnight Financing Rate) in %")
        #sofr_rate = sofr_rate_pre.replace("%", "")
    

    st.markdown("*Required fields")
    st.markdown("Note: Input fields highlighted in blue are sourced from Equifax for use in the model")
    

    if st.button('Connect & Save to Database'):
        conn = get_connection()
        if conn:
            st.toast("Successfully connected to the database !") 
            time.sleep(3)
            st.divider()
            # cursor = conn.cursor()
            insert_query = """INSERT INTO users (Average_Age_of_Secured_Trades, Number_of_30_days_Past_Due_Delinquencies, Number_of_Open_Loans, Number_of_Revolving_Accounts_with_Derogatory_Reports,
                            Total_Balance_on_Trades_Reported_in_6_Months, Debt_to_Income_Ratio, KYC_AML_Varified_as_Clear, Total_Liquidity, Verified_Liquidity_Coverage_3_Years, Loan_Amount, Stated_Net_Worth,
                            Verified_Net_Worth, Verified_Liabilities, Most_Recent_AGI, Credit_Score_of_Applicant) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            data = (Average_Age_of_Secured_Trades, Number_of_30_DPD_Delinquency_Occurrences_on_Trades_in_24_Months, Number_of_Open_Trades, Number_of_Revolving_Trades_with_Unpaid_Major_Derogatory_Reported_in_24_Months, 
                    Total_Balance_on_Trades_Reported_in_6_Months, Monthly_Disposable_Income_w_Current_Debt, kyc_or_aml_verified_as_clear, total_liquidity, verified_liquidity_coverage_3_years, 
                    loan_amount, stated_net_worth, verified_net_worth, verified_liabilities, most_recent_agi, credit_score_of_sponsors_1)
            conn.execute(insert_query, data)
            conn.commit()
            # cursor.close()
            conn.close()
            st.toast("Successfully saved to the database !") 
            time.sleep(3)

    if st.button('Credit Status'):
        credit_status = credit_status_prediction([Average_Age_of_Secured_Trades, Number_of_30_DPD_Delinquency_Occurrences_on_Trades_in_24_Months, Number_of_Open_Trades, 
                                                Number_of_Revolving_Trades_with_Unpaid_Major_Derogatory_Reported_in_24_Months, Total_Balance_on_Trades_Reported_in_6_Months, Monthly_Disposable_Income_w_Current_Debt], 
                                                kyc_or_aml_verified_as_clear, verified_liquidity_coverage_3_years, total_liquidity)

    st.success(credit_status)


if __name__ == '__main__':
    main()

   

# with tab2:
#     st.header("Example Applicant Playground")
#     sliders = []
#     col1, col2 = st.columns(2)
#     with col1:
#         for attribute in model_df.columns[:6]:
#             display_attribute = attribute.replace("_", " ")
#             ing_slider = st.slider(label=display_attribute, min_value=0, max_value=int(model_df[attribute].max()))
#             sliders.append(ing_slider)

#     with col2:
#         col1, col2 = st.columns(2, gap="medium")

#         target_names = np.array(['Non-Default', 'Default'])
        
#         prediction = lr_classif.predict([sliders])
#         probability = lr_classif.predict_proba([sliders])[0, 1]
#         # probability = round(probs[prediction[0]], 5)
#         with col1:
#             st.markdown("### Probability of Default : <strong style='color:tomato;'>{}</strong>".format("{:.2f} %".format((probability)*100)), unsafe_allow_html=True)

#         with col2:
#            if (prediction[0]==0):
#               if (probability) > (spread.iat[4,2] / 100):
#                  st.markdown('Loan should not be approved')
#               if (probability) <= (spread.iat[4,2] / 100) and (probability) > (spread.iat[3,2] / 100):
#                  st.markdown("### Rate Spread Applied : <strong style='color:tomato;'>{}</strong>".format("{:.2f} %".format(spread.iat[4,3])), unsafe_allow_html=True)
#               if (probability) <= (spread.iat[3,2] / 100) and (probability) > (spread.iat[2,2] / 100):
#                  st.markdown("### Rate Spread Applied : <strong style='color:tomato;'>{}</strong>".format("{:.2f} %".format(spread.iat[3,3])), unsafe_allow_html=True) 
#               if (probability) <= (spread.iat[2,2] / 100) and (probability) > (spread.iat[1,2] / 100):
#                  st.markdown("### Rate Spread Applied : <strong style='color:tomato;'>{}</strong>".format("{:.2f} %".format(spread.iat[2,3])), unsafe_allow_html=True)
#               if (probability) <= (spread.iat[1,2] / 100) and (probability) > (spread.iat[0,2] / 100):
#                  st.markdown("### Rate Spread Applied : <strong style='color:tomato;'>{}</strong>".format("{:.2f} %".format(spread.iat[1,3])), unsafe_allow_html=True)
#               elif (probability) <= (spread.iat[0,2] / 100):
#                  st.markdown("### Rate Spread Applied : <strong style='color:tomato;'>{}</strong>".format("{:.2f} %".format(spread.iat[0,2])), unsafe_allow_html=True)
#            elif (prediction[0]==1):
#               st.markdown('Loan should not be approved')           

         
# with tab3:
#     st.header("Confusion Matrix | Feature Importances")
#     col1, col2 = st.columns(2)
#     st.markdown("0: % Non-Defaults, 1: % Defaults")
#     plt.figure(figsize=(7,7))
#     conf_mat_fig = plt.figure(figsize=(5, 5))
#     ax1 = conf_mat_fig.add_subplot(111)
#     # skplt.metrics.plot_confusion_matrix(Y_test, Y_test_preds, figsize=(3,3), ax=ax1, normalize=True)
#     # Plotting ROC Curve using matplotlib
#     st.subheader("ROC Curve")
#     fpr, tpr, _ = roc_curve(y_test, y_probas[:, 1])
#     plt.figure(figsize=(10, 6))
#     plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc='lower right')
#     st.pyplot(plt)
#     plt.yticks(fontsize=7)
#     plt.xticks(fontsize=7)
#     st.pyplot(conf_mat_fig, use_container_width=True)

#     plt.figure(figsize=(21,16))
#     columns_names = ['Average Age \n of Secured Trades', 'Number of 30 DPD Delinquency \n Occurrences on Trades in 24 Months', 'Number of Open Trades', 'Number of Revolving Trades with Unpaid \n Major Derogatory Reported in 24 Months',
#                            'Total Balance on Trades \n Reported in 6 Months', 'Monthly Disposable Income \n w Current Debt']
#     feat_imp_fig = plt.figure(figsize=(6,6))
#     ax1 = feat_imp_fig.add_subplot(111)
#     coefficients = lr_classif.named_steps['model'].coef_[0].round(2)
#     feature_importance = pd.DataFrame({'Feature': columns_names, 'Importance': coefficients}) 
#     feature_importance = feature_importance.sort_values('Importance', ascending=True)
#     feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(28,37), ax=ax1)
#     plt.title("Feature Importances", fontsize=60)
#     plt.yticks(fontsize=30, weight='bold')
#     plt.xticks(fontsize=30, weight='bold')
#     plt.legend(fontsize=28)
#     plt.ylabel('xlabel', fontdict=None, labelpad=11)
#     ax1.bar_label(ax1.containers[0], size = 30, weight='bold')
#     st.pyplot(feat_imp_fig, use_container_width=True)
#     plt.tight_layout()
     
#     st.divider()
#     from sklearn.metrics import classification_report
#     report=classification_report(Y_test, Y_test_preds, output_dict=True)
#     df = pd.DataFrame(report).transpose()
#     def custom_format(val):
#         if val < 1:
#             return '{:.2f}'.format(val)
#         else:
#             return '{:.0f}'.format(val)
    
#     df = df.applymap(custom_format)
    
#     st.header("Classification Report")

#     # Convert dataframe to html and modify header alignment
#     html = df.to_html(classes='report-table', border = 0).replace('<th>', '<th style="text-indent: 0px;">')

#     # Embed the HTML in Streamlit
#     st.markdown(html, unsafe_allow_html=True)

#     st.markdown('<style>.report-table th { text-align: left; }</style>', unsafe_allow_html=True)

    
#     st.markdown("0: Non-Defaults")
#     st.markdown("1: Defaults")
#     st.markdown("Precision is a measure of how many of the positive predictions made are correct (true positives)")
#     st.markdown("Recall is a measure of how many of the positive cases the classifier correctly predicted, over all the positive cases in the data")
#     st.markdown("F1-Score is a measure combining both precision and recall. It is generally described as the harmonic mean of the two")
#     st.markdown("Support is the number of actual occurrences of the class in the specified dataset")


#     st.divider()

#     st.header("Model Dataset")
#     st.write(model_df)

# with tab4:
#     st.header("Risk Appetite and Spread Distribution")
    
#     st.write(spread.set_index('ID'))



