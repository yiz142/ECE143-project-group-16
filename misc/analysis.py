import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_hist(df, column):
    '''
    Draw a histogram of a given symptom.
    '''
    assert isinstance(df, pd.DataFrame)
    assert column in df.columns

    fig = plt.figure()
    fig = df[column].hist()
    fig.set_title(f"{column} distribution")
    fig.set_xlabel(f"{column}")
    fig.set_ylabel("count")


def draw_pie(df, column):
    '''
    Draw a pie-graph to show components of a given column.
    '''
    assert isinstance(df, pd.DataFrame)
    assert column in df.columns

    plt.figure(figsize=(6,6))
    plt.title(f"Composition of {column}")

    effects = df[column].value_counts()
    effects.plot(kind='pie',
                 colors=["green", "gold", "red"],
                 ylabel='',
                 autopct='%1.1f%%',
                 startangle=0,
                 wedgeprops={'edgecolor': 'black'})
    plt.show()


def draw_box_effect_symptom(df, symptom):
    '''
    Draw a boxplot of all effects and a given symptom.
    '''
    assert isinstance(df, pd.DataFrame)
    assert symptom in ["Depression", "Anxiety", "OCD", "Insomnia"]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Music effects", y=symptom, data=df)
    plt.title(f"Self-Reported Music Effects on {symptom} Levels")
    plt.show()


def draw_scatter_symptom_otherCol(df, column):
    '''
    Draw a scatterplot of all symptoms and a gvien column.
    '''
    assert isinstance(df, pd.DataFrame)
    assert column in df.columns

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=column, y="Anxiety", data=df, label="Anxiety")
    sns.scatterplot(x=column, y="Depression", data=df, label="Depression")
    sns.scatterplot(x=column, y="Insomnia", data=df, label="Insomnia")
    sns.scatterplot(x=column, y="OCD", data=df, label="OCD")
    plt.title(f"{column} vs Symptom Level")
    plt.xlabel(column)
    plt.ylabel("Symptom Level")
    plt.legend()
    plt.show()


def draw_multihist_genre_symptom(df, symptom):
    '''
    Draw a multi-histogram of all genres and a given symptom.
    '''
    assert isinstance(df, pd.DataFrame)
    assert symptom in ["Depression", "Anxiety", "OCD", "Insomnia"]

    chart = [{genre : 0 for genre in df['Fav genre']} for _ in range(3)]
    count = 0
    
    for item in df[symptom]:
        c = int(item)
        if c < 4:
            chart[0][df['Fav genre'].iloc[count]] += 1
        elif c < 7:
            chart[1][df['Fav genre'].iloc[count]] += 1
        else:
           chart[2][df['Fav genre'].iloc[count]] += 1
        count += 1
    
    # List of dictionaries
    dicts = chart
    
    # Get the keys from the first dictionary (assuming all dictionaries have the same keys)
    keys = list(dicts[0].keys())
    
    # Extract values for each dictionary
    values = [[d[key] / sum(d.values()) for key in keys] for d in dicts]
    
    # Set the positions for the bars
    x = np.arange(len(keys)) * 4 # the label locations
    
    # Width of the bars
    width = 0.5  # Adjust this to change the width of the bars
    
    # Plot the bars for each dictionary
    fig, ax = plt.subplots(figsize = (15, 5))
    
    # Create the bars for each dictionary
    labels = ['minimal ' + symptom, 'mild ' + symptom, 'severe ' + symptom]
    bars = [ax.bar(x + i * width, values[i], width, label=labels[i]) for i in range(len(dicts))]
    
    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel('genre')
    ax.set_ylabel('percent')
    ax.set_title('relative occurance of favorite genre')
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.legend()
    
    # Show the plot
    plt.show()


def draw_correlation_symptom_genre(df):
    '''
    Draw a heatmap of correlation matrix between all symptoms and all genres.
    '''
    assert isinstance(df, pd.DataFrame)

    genre_columns = [col for col in df.columns if "Frequency" in col]
    mental_health_columns = ["Anxiety", "Depression", "Insomnia", "OCD"]

    plt.figure(figsize=(12, 8))
    corr = df[genre_columns + mental_health_columns].corr()
    sns.heatmap(corr.loc[genre_columns, mental_health_columns], annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Music Genres and Mental Health")
    plt.show()


def draw_correlation_symptom_symptom(df):
    '''
    Draw a heatmap correlation matrix between all symptoms and themselves.
    '''
    assert isinstance(df, pd.DataFrame)
    
    df_encoded = df['Hours per day']
    df_relevant = df[['Depression', 'Anxiety','Insomnia', 'OCD']]
    df_combined = pd.concat([df_relevant, df_encoded], axis=1)
    correlation_matrix = df_combined.corr()
    sns.heatmap(correlation_matrix[0:4],
                annot=True,
                cmap='YlOrRd',
                fmt=".1f",
                annot_kws={"size": 7})
    plt.title('Correlation among Mental Health Signifiers and Favorite Genre')
    plt.figure(figsize = (20,10))
    plt.show()


def summary_effect_symptoms(df):
    '''
    Show stats info between effect and all symptoms
    '''
    assert isinstance(df, pd.DataFrame)
    
    summary_stats = df.groupby("Music effects")[['Depression', 'Anxiety','Insomnia', 'OCD']].mean()
    try:
        # If running in Jupyter Notebook
        from IPython.display import display
        display(summary_stats)
    except ImportError:
        # If IPython is not available, fallback to print
        print(summary_stats)
