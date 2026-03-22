# Importing necessary libraries for data processing and plotting
from pandas import read_excel  # For reading Excel files
from matplotlib import pyplot, font_manager  # For plotting graphs
import seaborn  # For statistical plotting
# import jieba  # For Chinese text segmentation
import re  # For regular expressions
import numpy  # For numerical operations
# from wordcloud import WordCloud  # For generating word clouds
# from PIL import Image  # For image processing

# Reading the dataset from an Excel file
Data = read_excel("./CMDC_Dataset.xlsx")

# Separating good and bad reviews based on the 'class' column
good_reviews = Data[Data['class'] == 0]['text']  # Reviews classified as good (class 0)
bad_reviews = Data[Data['class'] == 1]['text']   # Reviews classified as bad (class 1)

# Counting the number of good and bad reviews
count = Data['class'].value_counts()

# Setting global configurations for plot appearance
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/times.ttf')
pyplot.rcParams['font.family'] = "Times New Roman"  # Set the font to Times New Roman
pyplot.rcParams['axes.spines.top'] = False  # Remove the top border of the plot
pyplot.rcParams['axes.spines.right'] = False  # Remove the right border of the plot
pyplot.rcParams['axes.titlesize'] = 20  # Adjust the size of the title font
pyplot.rcParams['axes.labelsize'] = 18  # Adjust the size of the axis labels
pyplot.rcParams['legend.fontsize'] = 18  # Adjust the size of the legend text
pyplot.rcParams['font.size'] = 18  # Adjust the overall font size

# Function to plot the review counts (positive vs negative)
def plot_review_counts(counts, labels, colors, title="Class Distribution", figsize=(6, 6)):
    """
    Creates a bar plot comparing the count of positive and negative reviews.

    Args:
    - counts (list or tuple): A list or tuple of review counts, e.g., [positive_count, negative_count].
    - labels (list or tuple): A list or tuple of labels for the counts.
    - colors (list or tuple): A list or tuple of colors for the bars.
    - title (str): Title of the plot (default is "Review Count Comparison").
    - figsize (tuple): Size of the figure (default is (6, 6)).
    """
    # Set the figure size
    pyplot.rcParams['figure.figsize'] = figsize
    
    # Create a bar plot
    bars = pyplot.bar(labels, counts, width=0.6, color=colors)
    
    # Adding labels to the axes
    pyplot.ylabel('Number of Samples')
    pyplot.xlabel('Class')
    pyplot.title(title)
    
    # Displaying the count values on top of the bars
    for bar in bars:
        height = bar.get_height()  # Get the height (count) of the current bar
        pyplot.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')  # Position text above the bar
    
    # Save the plot as a PNG file
    pyplot.savefig('Review_Count_Comparison.pdf', dpi=600, bbox_inches='tight')
    
    # Display the plot on the screen
    pyplot.show()

# Example usage for the review count plot
positive_count = len(good_reviews)  # Count of good reviews
negative_count = len(bad_reviews)   # Count of bad reviews

# Plot the comparison of positive vs negative review counts
plot_review_counts([positive_count, negative_count], labels=["Healthy Control", "Depression"], colors=['green', 'red'])

# Function to calculate word count in each review
def calculate_word_count(review):
    """
    Calculates the word count of a review.

    Args:
    - review (str): A single review text.

    Returns:
    - int: The word count (number of characters) in the review.
    """
    return len(review.replace(" ", ""))  # Count all non-space characters

# Function to plot the word count distribution for positive reviews
def plot_positive_word_count_distribution(positive_counts, positive_title, plot_title):
    """
    Plots the word count distribution for positive reviews.

    Args:
    positive_counts (list or pandas.Series): Word counts of positive reviews.
    positive_title (str): Title for the positive review plot.
    plot_title (str): Overall plot title.
    """
    pyplot.figure(figsize=(6, 6))
    seaborn.histplot(positive_counts, color='blue', kde=True, stat='probability')
    pyplot.title(positive_title)
    pyplot.ylabel('Density')
    pyplot.xlabel('Text Length')
    pyplot.savefig(f'{plot_title}_positive.pdf', dpi=600, bbox_inches='tight')
    pyplot.show()

# Function to plot the word count distribution for negative reviews
def plot_negative_word_count_distribution(negative_counts, negative_title, plot_title):
    """
    Plots the word count distribution for negative reviews.

    Args:
    negative_counts (list or pandas.Series): Word counts of negative reviews.
    negative_title (str): Title for the negative review plot.
    plot_title (str): Overall plot title.
    """
    pyplot.figure(figsize=(6, 6))
    seaborn.histplot(negative_counts, color='red', kde=True, stat='probability')
    pyplot.title(negative_title)
    pyplot.ylabel('Density')
    pyplot.xlabel('Text Length')
    pyplot.savefig(f'{plot_title}_negative.pdf', dpi=600, bbox_inches='tight')
    pyplot.show()

# Calculate the word counts for positive and negative reviews
positive_word_counts = good_reviews.apply(calculate_word_count)
negative_word_counts = bad_reviews.apply(calculate_word_count)

# Display the word counts for both positive and negative reviews
print(f"Positive Review Word Counts:\n{positive_word_counts}")
print(f"\nNegative Review Word Counts:\n{negative_word_counts}")

# Plot the word count distribution for both positive and negative reviews
plot_positive_word_count_distribution(positive_word_counts, "Healthy-Control Text-Length Distribution", "Reviews Word Count Analysis")

plot_negative_word_count_distribution(negative_word_counts, "Depression Text-Length Distribution", "Reviews Word Count Analysis")

# Function to plot word count distribution for all reviews
def plot_word_count_distribution_all(word_counts, plot_title):
    """
    Plots the word count distribution for all reviews.

    Args:
    - word_counts (list or pandas.Series): Word counts for all reviews.
    - plot_title (str): Overall plot title.
    """
    # Create a plot for word count distribution
    pyplot.figure(figsize=(6, 6))
    seaborn.histplot(word_counts, kde=True, stat='probability')
    pyplot.title(plot_title)
    pyplot.ylabel('Density')
    pyplot.xlabel('Text Length')
    
    # Save the plot as a PNG file
    pyplot.savefig(f'{plot_title}.pdf', dpi=600, bbox_inches='tight')
    pyplot.show()

# Calculate the word count for all reviews (both positive and negative)
word_counts = Data['text'].apply(calculate_word_count)

# Display the word counts for all reviews
print(f"Review Word Counts:\n{word_counts}")

# Plot the word count distribution for all reviews
plot_word_count_distribution_all(word_counts, "Overall Text-Length Distribution")

# # Function to tokenize input text using Jieba and remove stopwords
# def tokenize_text(text, stop_words, cut_mode='accurate'):
#     """
#     Tokenizes input text using Jieba with a specified cutting mode.
    
#     Parameters:
#     - text: Input text string.
#     - stop_words: List of words to exclude.
#     - cut_mode: Cutting mode for Jieba, can be 'full', 'search', or 'accurate' (default).
    
#     Returns:
#     - A string with tokenized words, excluding stopwords.
#     """
#     if not isinstance(text, str):
#         raise ValueError("Input must be a string")
#     if not text.strip():
#         return ""
    
#     if cut_mode == 'full':
#         word_list = jieba.lcut(text, cut_all=True)
#     elif cut_mode == 'search':
#         word_list = jieba.lcut_for_search(text)
#     else:
#         word_list = jieba.lcut(text)
    
#     return " ".join(remove_non_chinese_characters("".join([word for word in word_list if word not in stop_words])))

# def load_stopwords(file_path):
#     """
#     Loads stopwords from a file and returns a list of them.
    
#     Parameters:
#     - file_path: Path to the stopwords file.
    
#     Returns:
#     - A list of stopwords.
#     """
#     with open(file_path, 'r', encoding='utf-8') as file:
#         stopwords = file.read().splitlines()
#     return [word.strip() for word in stopwords if word.strip()]

# def remove_non_chinese_characters(text):
#     """
#     Removes all characters from the string that are not Chinese characters.
    
#     Args:
#     - text (str): The input string.
    
#     Returns:
#     - str: The string with all non-Chinese characters removed.
#     """
#     # Regular expression to match Chinese characters (Han characters)
#     return re.sub(r'[^\u4e00-\u9fa5]', '', text)

# # Load stopwords list
# stopwords = load_stopwords("./Stopwords.txt")

# # Example stopwords (you can replace it with a file path for more stopwords)
# stopwords = ['的', '就是', '是']

# # Read Excel file again (if necessary for other operations)
# Data = read_excel("./Dataset.xlsx")  

# good_reviews = Data[Data['class'] == 0]['text']
# bad_reviews = Data[Data['class'] == 1]['text']

# # Apply tokenization and stopword removal
# good_reviews = good_reviews.apply(lambda text: tokenize_text(text, stopwords))
# bad_reviews = bad_reviews.apply(lambda text: tokenize_text(text, stopwords))

# # Wordcloud generation for good reviews
# font = r'C:\Windows\Fonts\STXINGKA.TTF'  # Set the font for Chinese characters
# mask_image = numpy.array(Image.open('cat-2074514_1920.jpg'))  # Mask image for word cloud

# # Generate and save word cloud for good reviews
# wordcloud = WordCloud(font_path=font, width=800, height=400, background_color="white", mask=mask_image).generate(good_reviews.to_string())
# wordcloud.to_file("wordcloud_goodviews.pdf")

# # Display the word cloud for good reviews
# pyplot.figure(figsize=(6, 6))
# pyplot.imshow(wordcloud, interpolation='bilinear')
# pyplot.axis("off")
# pyplot.show()

# # Generate and save word cloud for bad reviews
# wordcloud = WordCloud(font_path=font, width=800, height=400, background_color="white", mask=mask_image).generate(bad_reviews.to_string())
# wordcloud.to_file("wordcloud_badviews.pdf")

# # Display the word cloud for bad reviews
# pyplot.figure(figsize=(6, 6))
# pyplot.imshow(wordcloud, interpolation='bilinear')
# pyplot.axis("off")
# pyplot.show()
