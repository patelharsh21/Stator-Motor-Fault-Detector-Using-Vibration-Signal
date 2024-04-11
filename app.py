import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np
import autokeras as ak
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap 
from sklearn.metrics import f1_score,confusion_matrix, accuracy_score, precision_score, recall_score
import lime #LIME package
import lime.lime_tabular #the type of LIIME analysis we’ll do
from sklearn.metrics import ConfusionMatrixDisplay


#############################################################################################################################################
def main():
    # Register your pages
    pages = {
        "Home": home,
        "Exploratory data analysis":EDA,
        "Feature Selection":feature_selection,
        "Model Training ML model": Training,
        "Prediction using ML models":prediction,
        "Model Training using DL models": Training_keras,
        "Prediction using DL models":prediction_keras,
        "Explainable AI":XAI
    }

    st.sidebar.title("App with pages")

    # Widget to select your page, you can choose between radio buttons or a selectbox
    # page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page]()

state=st.session_state

#initializing the session state variables
if 'uploaded_file0' not in st.session_state:
   st.session_state.uploaded_file0=None
if 'uploaded_file25' not in st.session_state:
   st.session_state.uploaded_file25=None
if 'uploaded_file50' not in st.session_state:
   st.session_state.uploaded_file50=None
if 'uploaded_file75' not in st.session_state:
   st.session_state.uploaded_file75=None
   
if "df_0_features" not in st.session_state:
   st.session_state.df_0_features=None
if "df_25_features" not in st.session_state:
   st.session_state.df_25_features=None
if "df_50_features" not in st.session_state:
   st.session_state.df_50_features=None
if "df_75_features" not in st.session_state:
   st.session_state.df_75_features=None

if "df_0_features" not in st.session_state:
   st.session_state.df_0_featuresf=None
if "df_25_features" not in st.session_state:
   st.session_state.df_25_featuresf=None
if "df_50_features" not in st.session_state:
   st.session_state.df_50_featuresf=None
if "df_75_features" not in st.session_state:
   st.session_state.df_75_featuresf=None
if "concatenated_dataset" not in st.session_state:
   st.session_state.concatenated_dataset=None

def home():
    # Title and description
    st.title("Stator Motor Fault Detection")
    st.write("Welcome to our project: Stator Motor Fault Detection Using Vibration Signals. Upload your vibration data to check for fault severity.")
    
    # Upload button
    # st.session_state.uploaded_file0=st.file_uploader("Upload Vibration Data (CSV format) for normal case")
    # st.session_state.uploaded_file25=st.file_uploader("Upload Vibration Data (CSV format) for 25% severity case")
    # st.session_state.uploaded_file50=st.file_uploader("Upload Vibration Data (CSV format) for 50% severity case")
    # st.session_state.uploaded_file75=st.file_uploader("Upload Vibration Data (CSV format) for 75% severity case")
    
  

#######################################################################################################################################################

def extract_labels(labels, window_size=100, overlap=50):
   
    labels_list = []

    data_len = len(labels)

    for i in range(0, data_len - window_size + 1, overlap):
       
        window_labels = labels[i:i+window_size]

        # Extracting the label from the first element in the window
        label = window_labels.iloc[0]
        
        
        labels_list.append(label)
    labels_df = pd.DataFrame(labels_list, columns=['Label'])

    return labels_df

def extract_time_domain_features(data, window_size=100, overlap=50):
    features = []
    data_len = len(data)

    for i in range(0, data_len - window_size + 1, overlap):
        window = data[i:i+window_size]
        max_val = window.max()
        min_val = window.min()
        min_max_diff = max_val - min_val
        mean_val = window.mean()
        median_val = window.median()
        std_dev = window.std()
        skewness = skew(window)
        kurt = kurtosis(window)
        features.append([max_val, min_val, min_max_diff, mean_val, median_val, std_dev, skewness, kurt])

    return pd.DataFrame(features, columns=['Max', 'Min', 'Min-max Difference', 'Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'])


def extract_frequency_domain_features(data, window_size=100, overlap=50, sampling_rate=1000):
    features = []
    data_len = len(data)

    for i in range(0, data_len - window_size + 1, overlap):
        window = data[i:i + window_size]
        fft_values = np.fft.fft(window)
        fft_magnitude = np.abs(fft_values)
        fft_frequency = np.fft.fftfreq(len(fft_values), 1 / sampling_rate)
        peak_frequency = fft_frequency[np.argmax(fft_magnitude)]
        power_spectrum = np.square(fft_magnitude)
        spectral_entropy = -np.sum((power_spectrum / np.sum(power_spectrum)) * np.log2(power_spectrum / np.sum(power_spectrum)))
        features.append([peak_frequency, spectral_entropy])

    return pd.DataFrame(features, columns=['Peak Frequency', 'Spectral Entropy'])

###########################################################################################################################################################   
def EDA():
# Load the data and st.session_state.uploaded_file25 is not None and st.session_state.uploaded_file50 is not None and st.session_state.uploaded_file75 is not None
    if st.session_state.uploaded_file0 is None:
            # st.session_state.df0 = pd.read_csv(st.session_state.uploaded_file0)
            # st.session_state.df25 = pd.read_csv(st.session_state.uploaded_file25)
            # st.session_state.df50 = pd.read_csv(st.session_state.uploaded_file50)
            # st.session_state.df75 = pd.read_csv(st.session_state.uploaded_file75)
            state.df0 = pd.read_csv('/home/harsh/github/Stator-Fault-Diagnosis-in-Induction-Motor-using-Vibration-Signal--main/Normal.csv')
            state.df25 = pd.read_csv('/home/harsh/github/Stator-Fault-Diagnosis-in-Induction-Motor-using-Vibration-Signal--main/PP_25.csv')
            state.df50 = pd.read_csv('/home/harsh/github/Stator-Fault-Diagnosis-in-Induction-Motor-using-Vibration-Signal--main/PP_50.csv')
            state.df75 = pd.read_csv('/home/harsh/github/Stator-Fault-Diagnosis-in-Induction-Motor-using-Vibration-Signal--main/PP_75.csv')
            # st.session_state.df25 = pd.read_csv(st.session_state.uploaded_file25)
            # st.session_state.df50 = pd.read_csv(st.session_state.uploaded_file50)
            # st.session_state.df75 = pd.read_csv(st.session_state.uploaded_file75)
            st.session_state.df_0_features = extract_time_domain_features(st.session_state.df0['Acc'])
            st.session_state.df_25_features = extract_time_domain_features(st.session_state.df25['Acc'])
            st.session_state.df_50_features = extract_time_domain_features(st.session_state.df50['Acc'])
            st.session_state.df_75_features = extract_time_domain_features(st.session_state.df75['Acc'])

            st.session_state.df_0_featuresf = extract_frequency_domain_features(st.session_state.df0['Acc'])
            st.session_state.df_25_featuresf = extract_frequency_domain_features(st.session_state.df25['Acc'])
            st.session_state.df_50_featuresf= extract_frequency_domain_features(st.session_state.df50['Acc'])
            st.session_state.df_75_featuresf = extract_frequency_domain_features(st.session_state.df75['Acc'])

            
            
            # st.dataframe(st.session_state.df_0_features )
            # Streamlit app
            st.title("Time-Domain Feature Comparison")
            # Iterate through feature columns 
            feature_columns =  st.session_state.df_0_features.columns[:] # Exclude the 'Index' column  Index(['Min', 'Min-max Difference', 'Mean', 'Median', 'Std Dev', 'Skewness','Kurtosis'], dtype='object')
            for selected_feature in feature_columns:
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(st.session_state.df_0_features.index, st.session_state.df_0_features[selected_feature], label='Normal')
                ax.plot(st.session_state.df_25_features.index, st.session_state.df_25_features[selected_feature], label='25% Fault')
                ax.plot(st.session_state.df_50_features.index, st.session_state.df_50_features[selected_feature], label='50% Fault')
                ax.plot(st.session_state.df_75_features.index, st.session_state.df_75_features[selected_feature], label='75% Fault')

                plt.legend(['Normal', '25% Fault', '50% Fault', '75% Fault'])
                plt.xlabel("Date-Time")
                plt.ylabel(selected_feature)
                plt.title(f"{selected_feature} Comparison")

                # Display the plot in Streamlit
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(fig)
            st.title("Frequency-Domain Feature Comparison")
            feature_columnsf =  st.session_state.df_0_featuresf.columns[:] # Exclude the 'Index' column  Index(['Min', 'Min-max Difference', 'Mean', 'Median', 'Std Dev', 'Skewness','Kurtosis'], dtype='object')
            for selected_feature in feature_columnsf:
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(st.session_state.df_0_featuresf.index, st.session_state.df_0_featuresf[selected_feature], label='Normal')
                ax.plot(st.session_state.df_25_featuresf.index, st.session_state.df_25_featuresf[selected_feature], label='25% Fault')
                ax.plot(st.session_state.df_50_featuresf.index, st.session_state.df_50_featuresf[selected_feature], label='50% Fault')
                ax.plot(st.session_state.df_75_featuresf.index, st.session_state.df_75_featuresf[selected_feature], label='75% Fault')

                plt.legend(['Normal', '25% Fault', '50% Fault', '75% Fault'])
                plt.xlabel("Date-Time")
                plt.ylabel(selected_feature)
                plt.title(f"{selected_feature} Comparison")

                # Display the plot in Streamlit
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(fig)

            ### merging the datasets 
            
            st.session_state.df_0_features['Label'] = 0
            st.session_state.df_25_features['Label'] = 25
            st.session_state.df_50_features['Label'] = 50
            st.session_state.df_75_features['Label'] = 75

            concatenated_datasetFrequency = pd.concat([
            st.session_state.df_0_featuresf,
            st.session_state.df_25_featuresf,
            st.session_state.df_50_featuresf,
            st.session_state.df_75_featuresf
            ], ignore_index=True)  # Reset index for the concatenated DataFrame
         
            st.session_state.concatenated_dataset=pd.concat([
                st.session_state.df_0_features,
                st.session_state.df_25_features,
                st.session_state.df_50_features,
                st.session_state.df_75_features
            ], ignore_index=True)  # Reset index for the concatenated DataFrame
            
            st.session_state.concatenated_dataset["Peak Frequency"]=concatenated_datasetFrequency["Peak Frequency"]
            st.session_state.concatenated_dataset['Spectral Entropy']=concatenated_datasetFrequency['Spectral Entropy']

    else :
            st.write("Upload the Dataset to get the Exploratory Data Analysis")
##################################################################################################################################
def feature_selection_util(data, target_column, threshold=0.5):
    correlation = data.corr()[target_column]
    selected_features = correlation[abs(correlation) > threshold*0.01].index.tolist()
    return selected_features

def feature_selection():
    # Streamlit app
    st.title("Feature Selection with Slider")

    # Adjust the correlation threshold using a slider
    threshold = st.slider("Select Correlation Threshold")

    # Display the selected features based on the chosen threshold
    st.session_state.selected_features = feature_selection_util(st.session_state.concatenated_dataset, 'Label', threshold)
    st.write("Selected Features:", st.session_state.selected_features)
    
##################################################################################################################################
if "trained_models" not in st.session_state:
    st.session_state.trained_models ={}
def Training():
    # Load the concatenated dataset
    # Add a 'Label' column with values corresponding to each DataFrame

    # Concatenate the DataFrames
    st.session_state.data=st.session_state.concatenated_dataset[st.session_state.selected_features]
    # Split the dataset into features (X) and labels (y)
    X = st.session_state.data.drop('Label', axis=1)
    y = st.session_state.data['Label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of machine learning models to try
    models = {
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    # Create an empty list to store trained models
    

    # Create a Streamlit web app
    st.title("Machine Learning Model Accuracy")

   

    # Loop through each model
    for model_name, model in models.items():
        st.subheader(f"Model: {model_name}")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate and display accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Save the selected model to a file if it matches the user's selection
        # if model_name == selected_model:
        #     joblib.dump(model, f"{selected_model}_model.pkl")
        #     st.write(f"{selected_model} model saved!")

        # Append the trained model to the list
        st.session_state.trained_models[model_name]=model

    # Provide a link to download the saved model
    # Streamlit select box for model selection
    




from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def prediction():
    # Input the dataset to make predictions
    st.session_state.uploaded_file_prediction = st.file_uploader("Upload Vibration Data (CSV format) for your motor")
    selected_option = st.selectbox("Select a model to use for prediction", st.session_state.trained_models.keys())

    if st.session_state.uploaded_file_prediction is not None:
        # Extract features from the dataset
        st.session_state.dfprediction = pd.read_csv(st.session_state.uploaded_file_prediction)
        st.session_state.df_prediction_features = extract_time_domain_features(st.session_state.dfprediction['Acc'])
        st.session_state.df_prediction_featuresf = extract_frequency_domain_features(st.session_state.dfprediction['Acc'])
        st.session_state.df_prediction_features['Peak Frequency'] = st.session_state.df_prediction_featuresf['Peak Frequency']
        st.session_state.df_prediction_features['Spectral Entropy'] = st.session_state.df_prediction_featuresf['Spectral Entropy']
        
        if "Label" in st.session_state.selected_features:
            st.session_state.selected_features.remove("Label")

        # Select only the columns present in selected_features from the DataFrame
        st.session_state.df_prediction_features = st.session_state.df_prediction_features[st.session_state.selected_features]

        # Get the prediction array
        y_true = extract_labels(st.session_state.dfprediction['Label'])
        y_pred = st.session_state.trained_models[selected_option].predict(st.session_state.df_prediction_features)

        # Debugging print statements
        print("Length of y_true:", len(y_true))
        print("Length of y_pred:", len(y_pred))

        # Calculate and display confusion matrix, accuracy, precision, and recall
        confusion_mat = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        st.subheader("Confusion Matrix:")
        # st.write(confusion_mat)
        
        class_names = ['drop', 'allow', 'deny', 'reset-both']
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_names)
        disp.plot()
        st.pyplot()
        st.subheader("Accuracy:")
        st.write(f"{accuracy:.2%}")

        st.subheader("Precision:")
        st.write(f"{precision:.2%}")

        st.subheader("Recall:")
        st.write(f"{recall:.2%}")

        st.subheader("F1 score:")
        st.write(f"{f1:.2%}")

    else:
        st.write("Upload the dataset to get the predictions and evaluation metrics.")

def Training_keras():
    st.title("Deep Learning Model Training")

    # Load the concatenated dataset
    # Add a 'Label' column with values corresponding to each DataFrame

    # Concatenate the DataFrames
    if "Label" not in st.session_state.selected_features:
        st.session_state.selected_features.append("Label")
    st.session_state.data = st.session_state.concatenated_dataset[st.session_state.selected_features]

    # Split the dataset into features (X) and labels (y)
    X = st.session_state.data.drop('Label', axis=1)
    y = st.session_state.data['Label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("Train AutoKeras Model"):
        # Initialize AutoKeras Classifier
        clf = ak.StructuredDataClassifier(max_trials=20)  # You can adjust max_trials based on your computational resources

        # Search for the best model architecture and hyperparameters
        history = clf.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.2)  # Adjust epochs as needed

        # Output the specs of the best model
        best_trial = clf.tuner.oracle.get_best_trials(1)[0]  # Get the best trial
        best_model_spec = best_trial.hyperparameters.values
        st.write("Specifications of the Best AutoKeras Model:")
        st.write(best_model_spec)
          
        # Evaluate the model on the test set
        accuracy = clf.evaluate(X_test, y_test)

        st.write("accuracy")
        st.write(accuracy[1])

        # Save the trained AutoKeras model
        clf.export_model().save("autokeras_model", save_format="tf")
        st.write("AutoKeras model saved!")
        st.session_state.trained_model = clf  # Store the trained model

        # Create a line chart using Matplotlib
        fig, ax = plt.subplots()
        epochs = range(1, len(history.history['loss']) + 1)
        ax.plot(epochs, history.history['loss'])

        # Set x and y labels
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        # Display the plot using Streamlit
        st.pyplot(fig)


def prediction_keras():
    if st.session_state.trained_model is None:
        st.warning("You need to train the AutoKeras model first.")
        return

    # Input the dataset to do prediction
    st.session_state.uploaded_file_prediction = st.file_uploader("Upload Vibration Data (CSV format) for your motor")
    if st.session_state.uploaded_file_prediction is not None:
        # Load the AutoKeras model
        model = st.session_state.trained_model

        # Extract features from the dataset
        st.session_state.dfprediction = pd.read_csv(st.session_state.uploaded_file_prediction)
        st.session_state.df_prediction_features = extract_time_domain_features(st.session_state.dfprediction['Acc'])
        st.session_state.df_prediction_featuresf = extract_frequency_domain_features(st.session_state.dfprediction['Acc'])
        st.session_state.df_prediction_features['Peak Frequency'] = st.session_state.df_prediction_featuresf['Peak Frequency']
        st.session_state.df_prediction_features['Spectral Entropy'] = st.session_state.df_prediction_featuresf['Spectral Entropy']
    
        if "Label" in st.session_state.selected_features:
            st.session_state.selected_features.remove("Label")

        # Select only the columns present in selected_features from the DataFrame
        st.session_state.df_prediction_features = st.session_state.df_prediction_features[st.session_state.selected_features]

        y_true = extract_labels(st.session_state.dfprediction['Label'])
        
        # Predict probabilities for each class
        y_probabilities = model.predict(st.session_state.df_prediction_features)

        # # Find the class with the highest probability for each sample
        # y_pred = np.argmax(y_probabilities, axis=1)
        # print(y_probabilities)
        # Debugging print statements
        # Convert string probabilities to float
        y_probabilities_float = y_probabilities.astype(float)

        # Convert float probabilities to integers
        y_pred_int = (y_probabilities_float + 0.5).astype(int)
        print("Length of y_true:", y_true.values[0])
        print("Length of y_pred:",y_pred_int[0])
        y_pred=y_pred_int

        # Calculate and display confusion matrix, accuracy, precision, recall, and F1 score
        confusion_mat = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        class_names = ['drop', 'allow', 'deny', 'reset-both']
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_names)
        disp.plot()
        st.pyplot()

        st.subheader("Accuracy:")
        st.write(f"{accuracy:.2%}")

        st.subheader("Precision:")
        st.write(f"{precision:.2%}")

        st.subheader("Recall:")
        st.write(f"{recall:.2%}")

        st.subheader("F1 Score:")
        st.write(f"{f1:.2%}")



def XAI():
    # Split the dataset into features (X) and labels (y)
    X = st.session_state.data.drop('Label', axis=1)
    y = st.session_state.data['Label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # XGBoost Classifier
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(label_encoder.classes_))
    model.fit(X_train, y_train_encoded)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    st.title("Global Plots")
    st.write("1.Summary plot ( 0=normal, 1=25% severity, 2=50% severity, 3=75% severity)")
    
    fig1=shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, plot_type="bar")
    st.pyplot(fig1)
    st.write("2. Bar plot")
    
    # %matplotlib inline
    # Calculate mean SHAP values for each class
    shap_valuesnew = explainer(X_train)
    mean_0 = np.mean(np.abs(shap_valuesnew.values[:, :, 0]),axis=0)
    mean_1 = np.mean(np.abs(shap_valuesnew.values[:, :, 1]),axis=0)
    mean_2 = np.mean(np.abs(shap_valuesnew.values[:, :, 2]),axis=0)
    mean_3 = np.mean(np.abs(shap_valuesnew.values[:, :, 3]),axis=0)

    # Create a DataFrame with lists
    df = pd.DataFrame({"Normal": mean_0, "25% severity": mean_1, "50% severity": mean_2, "75% severity": mean_3})

    # Plot mean SHAP values
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    df.plot.bar(ax=ax)
    
    ax.set_ylabel("Mean SHAP", size=20)
    # Set the x-axis tick labels with the correct number of locations
    ax.set_xticklabels(X_train.columns, rotation=45, size=10)

    ax.legend(fontsize=10)

    # Display the plot
    st.pyplot()
    # # List of options
    # options = ['Normal', '25% severity', '50% severity' ,'75% severity']
    # dict={'Normal' :0, '25% severity':1, '50% severity':2,'75% severity':3}
    # # Create a select box with a default selection
    # st.write("3.decision plot")
    # selected_option = st.selectbox('Select an option:', options, index=1)  # Default to the second option
    # labelForDecisionPlot=(dict[selected_option])
    # fig2=shap.decision_plot(explainer.expected_value[0], shap_values[labelForDecisionPlot][0:50], feature_names = list(X_test.columns),ignore_warnings=True)
    # st.pyplot(fig2)
    
    st.title("Local Plots")
    st.write("1. Waterfall plot")
    user_input = st.number_input("Enter an Index of datapoint:", value=0, step=1)
    st.write("prediction=",y_train.iloc[user_input])

    # Cast the input to an integer
    index= int(user_input)
    label=np.argmax(shap_values[index][0][:])
    # shap.plots.waterfall(shap_valuesnew[index][:][label])
   
    row = index
   
    # shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
    #                                           base_values=explainer.expected_value[0], data=X_test.iloc[row],  
    #                                      feature_names=X_test.columns.tolist()))
    st.pyplot()
    # st.write("2.Lime explainer")
    # # Create a LIME explainer
    # explainerlime = lime.lime_tabular.LimeTabularExplainer(X_train.values)
    
    # # Explain a specific instance using LIME
    # expXGB = explainerlime.explain_instance(X_test.iloc[index], model.predict_proba, num_features=10)

    # # Convert LIME explanation to a Matplotlib figure
    # lime_plot = expXGB.as_pyplot_figure()

    # # Display the LIME plot in Streamlit
    # st.pyplot(lime_plot)



if __name__ == "__main__":
    main()


