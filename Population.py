import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    col1, col2, col3 = st.columns([1,3,1])
    #data cleaning and extracting relevant features
    col2.write("# Raw Data")
    df1 = pd.read_csv("Data/food_coded.csv")
    col2.write(df1)

    col2.write("# Cleaned Data")
    df=df1[["cook","diet_current_coded","eating_out","sports","exercise","fav_cuisine_coded","on_off_campus","pay_meal_out","fav_food","fruit_day","income"]]
    df.dropna(axis=0,inplace=True)
    df.to_csv("Data/food_choices.csv")
    col2.write(df)


    # Plotting Boxplot for cleaned data
    col2.write('# Plotting Boxplot for Cleaned Data')
    fig, ax = plt.subplots()
    sns.boxplot(data=df, palette="Set1", ax=ax)
    ax.tick_params(labelsize=8.7)
    plt.xticks(rotation=45, ha='right')

    # Display the plot using Streamlit
    col2.pyplot(fig)


    col2.write("# Elbow Plot for Optimal K")
    wcss = []

    # Fit K-means clustering for different numbers of clusters
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=60)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    # Plotting the elbow graph
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    ax.set_title('The Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')

    # Display the plot in Streamlit
    col2.pyplot(fig)


    col2.write("# Optimal K-means Clustering")
    #K-Means clustering on cleaned data
    k = 3
    kmeans = KMeans(n_clusters = k, random_state=0).fit(df)
    df['Cluster']=kmeans.labels_

    col2.write(df)

    col2.write("# Box Plot for Optimal K-Means")
    fig, axes = plt.subplots(1, k, sharey=True)
    axes[0].set_ylabel('Coded Values', fontsize=20)

    for i in range(k):
        plt.sca(axes[i])
        plt.xticks(rotation=45, ha='right')
        sns.boxplot(palette="Set1", data=df[df['Cluster'] == i].drop('Cluster', 1), ax=axes[i]).tick_params(labelsize=8.7)

    # Display the plot in Streamlit
    col2.pyplot(fig)


if __name__ == "__main__":
    main()