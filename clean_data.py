import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

class Clean_Data:
    def __init__(self, path_data = str):
        self.path_data = path_data
        self.uniq_letters = sorted(set.union(*list(self.getNouns()['das_wort'].apply(lambda x: set(x)))))

    def getPath(self):
        return self.path_data
    
    def getNouns(self) -> pd.DataFrame:
        """Loads data acquired from dict.cc
        Returns a pandas DataFrame with cols: | word | gender |
        and around 250k lines."""
        
        with open(self.path_data, 'r') as file:
            data = file.read()

        # Create a container for holding lines found in data
        lines = [i for i in data.split('\n')]

        # Keep lines which contain 3 tabs
        lines = [i for i in lines if i.count('\t') == 3]

        # Split the lines along tabs. The result is a 2d matrix.
        data_mat = [i.split('\t') for i in lines]

        # Build a dataframe
        df = pd.DataFrame(data_mat, columns=['german', 'english', 'part_of_speech', 'area'])

        # Create a new dataframe which contain nouns
        df_noun = df[df['part_of_speech'] == 'noun']

        # Filter the column where this pattern occurs:
        pattern = r"^[A-ZÄÖÜ][a-zäöüß]+\s\{[mfn]\}$"
        # Filter the data based on pattern.
        df_filt = df_noun[df_noun['german'].str.match(pattern)]

        # Get rid of duplicates
        df_filt = df_filt.drop_duplicates(subset=['german'])

        # Make it lowercase and get split it along spaces, the result is a 2D list
        data2 = [noun_str.lower().split(' ') for noun_str in df_filt['german']]
        # Create DF with two cols
        df_final = pd.DataFrame(data2, columns=['das_wort', 'die_artikel'])

        # Remove curly brackets
        df_final['die_artikel'] = df_final['die_artikel'].apply(lambda x: x.strip('{}')) # x[1] also work...
        return df_final
    
    def one_hot_encode(self, word = str) -> np.array:
        """Creates a one hot encoded version of a string of characters.
        One line of the output corresponds to one character of the input."""
        # Container to store letter representing vectors.
        word_encoded = list()
        
        # Loop through the word.
        for character in word:
            char_vec = np.where(np.array(self.uniq_letters) == character, 1.0, 0.0)
            word_encoded.append(char_vec)
        return np.array(word_encoded)

    def bag_of_letters(self, word = str) -> np.array:
        """Compact one hot encode."""
        return np.sum(self.one_hot_encode(word), axis = 0)

    def genderPiechart(self) -> None:
        """Display the distribution of masculine, feminine and neuter words on a piechart."""
        df_final = self.getNouns()
        # Display the distribution of words on a piechart by ChatGPT
        # Map labels and colors 
        label_map = {'m': 'der', 'f': 'die', 'n': 'das'}
        color_map = {'der': 'forestgreen', 'die': 'firebrick', 'das': 'darkorange'}
        # Count occurrences of each unique value in column 'die_artikel'
        value_counts = df_final['die_artikel'].map(label_map).value_counts()
        # Plot pie chart 
        plt.figure(figsize=(6, 6))
        plt.pie(
            value_counts,
            labels=value_counts.index,
            autopct='%1.1f%%',
            colors=[color_map[label] for label in value_counts.index],
            startangle=140
        )
        plt.title('Distribution of Gendered Articles')
        plt.show()
        