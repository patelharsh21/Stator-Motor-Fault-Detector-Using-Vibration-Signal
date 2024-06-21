import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np
import autokeras as ak
from sklearn.preprocessing import LabelEncoder
import shap 
from sklearn.metrics import f1_score,confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from tabulate import tabulate
import tensorflow as tf

from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler



## MAIN FUNCTION
def main():
    # Register your pages
    pages = {
        "Dataset Upload": home,
        "Exploratory data analysis" : EDA,
        "Feature Selection":feature_selection,
        "Model Training using DL models": Training_keras,
        "Prediction using DL models":prediction_keras,
        "Explainable AI":XAI
    }

    st.sidebar.title("Fault Diagnosis")

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

## FEATURE EXTRACTION FUNCTIONS

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

## FOR EXTRACTING THE LABELS FROM THE WINDOW 
## We take the label of the first datapoint of the window
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




def home():
    # Title and description
    st.title("Stator Motor Fault Detection")
    st.write("Welcome to our project: Stator Motor Fault Detection Using Vibration Signals. Upload your vibration data to check for fault severity.")
    
    # # Upload button
    # st.session_state.uploaded_file0=st.file_uploader("Upload Vibration Data (CSV format) for normal case")
    # st.session_state.uploaded_file25=st.file_uploader("Upload Vibration Data (CSV format) for 25% severity case")
    # st.session_state.uploaded_file50=st.file_uploader("Upload Vibration Data (CSV format) for 50% severity case")
    # st.session_state.uploaded_file75=st.file_uploader("Upload Vibration Data (CSV format) for 75% severity case")


def balance_and_normalize_data(data):
    # Balance the data
    class_counts = data['Label'].value_counts()
    min_class_count = class_counts.min()

    balanced_data = pd.concat([
        resample(data[data['Label'] == label], replace=False, n_samples=min_class_count, random_state=42)
        for label in class_counts.index
    ])

    # Normalize the features
    scaler= MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(balanced_data.drop('Label', axis=1)), columns=data.columns[:-1])
    normalized_data['Label'] = balanced_data['Label']

    return normalized_data


def EDA():
# Load the data and st.session_state.uploaded_file25 is not None and st.session_state.uploaded_file50 is not None and st.session_state.uploaded_file75 is not None
    if st.session_state.uploaded_file0 is None:
            # st.session_state.df0 = pd.read_csv(st.session_state.uploaded_file0)
            # st.session_state.df25 = pd.read_csv(st.session_state.uploaded_file25)
            # st.session_state.df50 = pd.read_csv(st.session_state.uploaded_file50)
            # st.session_state.df75 = pd.read_csv(st.session_state.uploaded_file75)

            state.df0 = pd.read_csv('./Normal.csv')
            state.df25 = pd.read_csv('./PP_25.csv')
            state.df50 = pd.read_csv('./PP_50.csv')
            state.df75 = pd.read_csv('./PP_75.csv')
            # st.session_state.df25 = pd.read_csv(st.session_state.uploaded_file25)
            # st.session_state.df50 = pd.read_csv(st.session_state.uploaded_file50)
            # st.session_state.df75 = pd.read_csv(st.session_state.uploaded_file75)
            state.df_0_features = extract_time_domain_features(state.df0['Acc'])
            state.df_25_features = extract_time_domain_features(state.df25['Acc'])
            state.df_50_features = extract_time_domain_features(state.df50['Acc'])
            state.df_75_features = extract_time_domain_features(state.df75['Acc'])

            state.df_0_featuresf = extract_frequency_domain_features(state.df0['Acc'])
            state.df_25_featuresf = extract_frequency_domain_features(state.df25['Acc'])
            state.df_50_featuresf= extract_frequency_domain_features(state.df50['Acc'])
            state.df_75_featuresf = extract_frequency_domain_features(state.df75['Acc'])

            
            
            # st.dataframe(st.session_state.df_0_features )
            # Streamlit app
            st.title("Time-Domain Feature Comparison")
            # Iterate through feature columns 
            feature_columns =  st.session_state.df_0_features.columns[:] # Exclude the 'Index' column  Index(['Min', 'Min-max Difference', 'Mean', 'Median', 'Std Dev', 'Skewness','Kurtosis'], dtype='object')
            for selected_feature in feature_columns:
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(state.df_0_features.index, state.df_0_features[selected_feature], label='Normal')
                ax.plot(state.df_25_features.index, state.df_25_features[selected_feature], label='25% Fault')
                ax.plot(state.df_50_features.index, state.df_50_features[selected_feature], label='50% Fault')
                ax.plot(state.df_75_features.index, state.df_75_features[selected_feature], label='75% Fault')

                plt.legend(['Normal', '25% Fault', '50% Fault', '75% Fault'])
                plt.xlabel("Date-Time")
                plt.ylabel(selected_feature)
                plt.title(f"{selected_feature} Comparison")

                # Display the plot in Streamlit
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(fig)
            st.title("Frequency-Domain Feature Comparison")
            feature_columnsf =  state.df_0_featuresf.columns[:] # Exclude the 'Index' column  Index(['Min', 'Min-max Difference', 'Mean', 'Median', 'Std Dev', 'Skewness','Kurtosis'], dtype='object')
            for selected_feature in feature_columnsf:
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(state.df_0_featuresf.index, state.df_0_featuresf[selected_feature], label='Normal')
                ax.plot(state.df_25_featuresf.index, state.df_25_featuresf[selected_feature], label='25% Fault')
                ax.plot(state.df_50_featuresf.index, state.df_50_featuresf[selected_feature], label='50% Fault')
                ax.plot(state.df_75_featuresf.index, state.df_75_featuresf[selected_feature], label='75% Fault')

                plt.legend(['Normal', '25% Fault', '50% Fault', '75% Fault'])
                plt.xlabel("Date-Time")
                plt.ylabel(selected_feature)
                plt.title(f"{selected_feature} Comparison")

                # Display the plot in Streamlit
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(fig)

            ### merging the datasets 
            
            state.df_0_featuresf['Label'] = 0
            state.df_25_featuresf['Label'] = 25
            state.df_50_featuresf['Label'] = 50
            state.df_75_featuresf['Label'] = 75

            concatenated_datasetFrequency = pd.concat([
            state.df_0_featuresf,
            state.df_25_featuresf,
            state.df_50_featuresf,
            state.df_75_featuresf
            ], ignore_index=True)  # Reset index for the concatenated DataFrame
         
            state.concatenated_dataset=pd.concat([
                state.df_0_features,
                state.df_25_features,
                state.df_50_features,
                state.df_75_features
            ], ignore_index=True)  # Reset index for the concatenated DataFrame
            
            state.concatenated_dataset["Peak Frequency"]=concatenated_datasetFrequency["Peak Frequency"]
            state.concatenated_dataset['Spectral Entropy']=concatenated_datasetFrequency['Spectral Entropy']
            state.concatenated_dataset['Label']=concatenated_datasetFrequency['Label']

    else :
            st.write("Upload the Dataset to get the Exploratory Data Analysis")

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

def count_trainable_parameters(model):
    trainable_count = int(np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights]))
    return trainable_count

def display_model_summary(model):
    model_summary = []
    for layer in model.layers:
        layer_info = {
            'Layer': layer.name,
            'Parameters': layer.count_params(),
            'Output Shape': layer.output_shape
        }
        model_summary.append(layer_info)
    return pd.DataFrame(model_summary)

def Training_keras():
    st.title("Deep Learning Model Training")

    # Load the concatenated dataset
    state.data = st.session_state.concatenated_dataset[st.session_state.selected_features]

    st.session_state.data = st.session_state.concatenated_dataset[st.session_state.selected_features]
    X = st.session_state.data.drop('Label', axis=1)
    y = st.session_state.data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("Train AutoKeras Model"):
        clf = ak.StructuredDataClassifier(max_trials=10)
        history = clf.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.2)

        # Get the best model architecture and hyperparameters
        best_trial = clf.tuner.oracle.get_best_trials(1)[0]
        best_model_spec = best_trial.hyperparameters.values

        # Extract the trained model
        model = clf.export_model()

        # Display model summary in Streamlit
        st.subheader("AutoKeras Model Summary:")
        model.summary(print_fn=st.text)

        # Evaluate the model on the test set
        accuracy = clf.evaluate(X_test, y_test)
        num_params = model.count_params()
        num_params_trainable = count_trainable_parameters(model)
        num_params_non_trainable = num_params - num_params_trainable

        st.write("Number of Parameters in the Model:", num_params)
        st.write("Number of Trainable Parameters in the Model:", num_params_trainable)
        st.write("Number of Non-Trainable Parameters in the Model:", num_params_non_trainable)

        st.write("Accuracy:", accuracy[1])
        st.write("Loss:", accuracy[0])

        # Save the trained AutoKeras model
        clf.export_model().save("autokeras_model", save_format="tf")
        st.write("AutoKeras model saved!")
        st.session_state.trained_model = clf  # Store the trained model

        # Plot training history with epoch vs. loss and accuracy
        st.subheader("Epoch vs. Loss and Accuracy Curves")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create two subplots

        epochs = range(1, len(history.history['loss']) + 1)

        major_font_size = 20
        minor_font_size = 18
        st.subheader("Epoch vs. Loss and Accuracy Curves")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create two subplots

        epochs = range(1, len(history.history['loss']) + 1)

        # Plot loss
        axs[0].plot(epochs, history.history['loss'], label='Loss')
        axs[0].set_xlabel('Epoch', fontsize=major_font_size)
        axs[0].set_ylabel('Loss', fontsize=major_font_size)
        axs[0].legend(fontsize=18)
        axs[0].tick_params(axis='both', which='major', labelsize=major_font_size)
        axs[0].tick_params(axis='both', which='minor', labelsize=minor_font_size)

        # Plot accuracy
        axs[1].plot(epochs, history.history['accuracy'], label='Accuracy')
        axs[1].set_xlabel('Epoch', fontsize=major_font_size)
        axs[1].set_ylabel('Accuracy', fontsize=major_font_size)
        axs[1].legend(fontsize=18)
        axs[1].tick_params(axis='both', which='major', labelsize=major_font_size)
        axs[1].tick_params(axis='both', which='minor', labelsize=minor_font_size)

        st.pyplot(fig)
# ... (rest of your code)

def prediction_keras():
    st.title("Prediction using Autokeras")
    if st.session_state.trained_model is None:
        st.warning("You need to train the AutoKeras model first.")

    # Input the dataset to do prediction
    # st.session_state.uploaded_file_prediction = st.file_uploader("Upload Vibration Data (CSV format) for your motor")
    st.session_state.uploaded_file_prediction =pd.read_csv('./testing.csv')
    if st.session_state.uploaded_file_prediction is not None:
        # Load the AutoKeras model
        model = st.session_state.trained_model
        # Extract features from the dataset
        state.dfprediction = state.uploaded_file_prediction 
        state.df_prediction_features = extract_time_domain_features(state.dfprediction['Acc'])
        state.df_prediction_featuresf = extract_frequency_domain_features(state.dfprediction['Acc'])
        state.df_prediction_features['Peak Frequency'] = state.df_prediction_featuresf['Peak Frequency']
        state.df_prediction_features['Spectral Entropy'] = state.df_prediction_featuresf['Spectral Entropy']
    
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
        y_pred=y_pred_int

        # Calculate and display confusion matrix, accuracy, precision, recall, and F1 score
        confusion_mat = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        class_names = ['Normal', '25% severity', '50% severity', '75% severity']
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_names)
        disp.plot()
        st.pyplot()

        st.subheader("Accuracy:")
        st.write(f"{accuracy:.2%}")
        
        st.subheader("Metrics table")
        # List of classes
        classes = [0,25,50,75]

        # Initialize empty lists to store metrics for each class
        precision_list = []
        recall_list = []
        f1_list = []

        # Calculate metrics for each class
        for label in classes:
            precision = precision_score(y_true, y_pred, labels=[label], average=None)[0]
            recall = recall_score(y_true, y_pred, labels=[label], average=None)[0]
            f1 = f1_score(y_true, y_pred, labels=[label], average=None)[0]

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        
        # for calculating wieghted average
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        classes.append("Wieghted avg")
        # Create a table
        table = {
            "Class": classes,
            "Precision": precision_list,
            "Recall": recall_list,
            "F1 Score": f1_list
        }

        # Print the table
        table = st.table(table)




import streamlit.components.v1 as components

# def st_shap(plot, height):
#     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>" 
#     components.html(shap_html, height=height,width=1200)

# def XAI():
#     X = st.session_state.data.drop('Label', axis=1)
#     y = st.session_state.data['Label']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    
#     st.title("Multi output shap force plot :")
#     explainer = shap.KernelExplainer( state.trained_model.export_model(),X_test[:25])
#     shap_values = explainer.shap_values(X_test[:25])
#     st_shap(shap.plots.force(explainer.expected_value[0], shap_values[0]),height=500)

#     st.title("Single output shap force plot :")
#     shap_values_ind = explainer.shap_values(X_train.iloc[25,:], nsamples=500)
#     st_shap(shap.plots.force(explainer.expected_value[0], shap_values_ind[0]),height=500)
    
#     st.title(" Bar plot for showing feature importance :")
#     # Assuming shap_values is a list of arrays for each class
#     mean_shap_values = np.mean([np.abs(class_values) for class_values in shap_values], axis=1)

#     # Create a DataFrame with lists
#     df = pd.DataFrame(mean_shap_values.T, columns=["Normal", "25% severity", "50% severity", "75% severity"])

#     # Plot mean SHAP values
#     fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#     df.plot.bar(ax=ax)

#     ax.set_ylabel("Mean SHAP", size=20)
#     # Set the x-axis tick labels with the correct number of locations
#     ax.set_xticklabels(X_train.columns, rotation=45, size=10)

#     ax.legend(fontsize=10)

#     # Display the plot
#     plt.show()
#     st.pyplot()
    


# Define the st_shap function
def st_shap(plot, height):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height, width=1200)

# Define the XAI function
def XAI():
    # Load your data and split it
    X = st.session_state.data.drop('Label', axis=1)
    y = st.session_state.data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    # st.title("Multi-output SHAP Force Plot:")
    explainer = shap.KernelExplainer(state.trained_model.export_model(), X_test[:25])
    shap_values = explainer.shap_values(X_test[:25])
    # st_shap(shap.plots.force(explainer.expected_value[0], shap_values[3]), height=500)    
    
    st.title("Global Explanation")
    st.markdown("**1. SHAP Summary Plot:**", unsafe_allow_html=True)
    class_names = ['Normal', '25% fault','50% fault','75% fault']  # Replace with your actual class names
    shap.summary_plot(shap_values, X_test[:25], feature_names=X_train.columns, class_names=class_names)
    # Customizing plot text
    plt.title('SHAP Summary Plot', fontsize=20, fontweight='bold')  # Title
    plt.xlabel('SHAP Value', fontsize=15, fontweight='bold')  # X-axis label
    plt.ylabel('Features', fontsize=15, fontweight='bold')  # Y-axis label
    plt.xticks(fontsize=12, fontweight='bold')  # X-axis ticks size
    plt.yticks(fontsize=12, fontweight='bold')  # Y-axis ticks size

    plt.tight_layout() 
    st.pyplot()


    st.markdown("**2. Bar Plot for Showing Feature Importance:**", unsafe_allow_html=True)
    # Assuming shap_values is a list of arrays for each class
    mean_shap_values = np.mean([np.abs(class_values) for class_values in shap_values], axis=1)

    # Create a DataFrame with lists
    df = pd.DataFrame(mean_shap_values.T, columns=["Normal", "25% severity", "50% severity", "75% severity"])

    # Plot mean SHAP values
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    df.plot.bar(ax=ax)
    
    ax.set_ylabel("Mean SHAP", size=30, weight='bold') # Make y-axis label bold
    ax.tick_params(axis='y', labelsize=30) # Increase size of y-axis tick labels
    ax.set_xticklabels(X_train.columns, rotation=45, size=30, weight='bold') # Make x-axis tick labels bold
    ax.legend(fontsize=30) # Make legend labels bold

    # Display the plot in Streamlit 
    st.pyplot()
    
    

    st.title("Local Explanation")
    st.markdown("**1.SHAP Force Plot:**", unsafe_allow_html=True)
    index = st.number_input("Enter an Index of datapoint:", value=0, step=1)
    
    shap_values_ind = explainer.shap_values(X_train.iloc[index, :], nsamples=500)
    # st.write(len(shap_values_ind))
    # st.write(len(shap_values_ind[0]))
    # st.write(len(shap_values_ind[0][0]))
    
    # first_feature_values = [item[0] for item in shap_values_ind]

    # # Find the label along which the value of the first feature is highest
    # label_with_highest_value = max(enumerate(first_feature_values), key=lambda x: x[1])[0]
    # Convert the list to a NumPy array for easier calculations

    shap_values_array = np.array(shap_values_ind)

    # Calculate the mean value along the second dimension (axis=1) because that will be the predicted label for the given datapoint
    mean_values = np.mean(shap_values_array, axis=1)

    # Find the index of the label with the highest mean value
    label = np.argmax(mean_values)
    mapping = {0: "Normal", 1: "25% severity", 2: "50% severity", 3: "75% severity"}

    st.write(mapping[label])

    # Retrieve the feature names
    feature_names = X_train.columns

    # Generate SHAP force plot for the selected label
    fig = shap.plots.force(explainer.expected_value[0], shap_values_ind[label], feature_names=feature_names)

    # Increase font size
    plt.rcParams.update({'font.size': 14})  # Adjust font size as needed

    st_shap(fig,height=200)

if __name__ == "__main__":
    main()
