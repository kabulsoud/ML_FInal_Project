import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load the dataset
df = pd.read_csv("Data Science job Postings - job_postings.csv")
# print(df.head())

## Preprocessing data
df['job_skills'] = df['job_skills'].fillna("").astype(str).str.lower().str.split(',').apply(tuple) # Lowercase, replace missing values, and split, then convert to tuple

## Handling missing or duplicate values
df = df.dropna(subset=['job_level', 'job_skills'])
df = df.drop_duplicates()

print("Data preprocessed successfully") ## Debugging 


## Job_Skills analysis
skills = df['job_skills'].explode()
skills_count = skills.value_counts().head(10) ## to 10 skills to visualize

# ## Visualizing to 10 skills from dataset
# plt.figure(figsize=(10,6))
# skills_count.plot(kind='bar', color = 'lightblue')
# plt.title("Visualizing top 10 Skills for Data Science Roles")
# plt.xlabel('Skills')
# plt.ylabel("Skill Frequency")
# plt.xticks(rotation=40)
# plt.tight_layout()
# plt.savefig("Top_10_Skills_Bar_Graph.png")
# plt.show()


## Using wordcloud to tell us whihch words, and thus what elements is used the most in Job_level whihc makes correlation between job level and skills required
## Extra visualization
# for levels in df["job_level"].unique():
#     levels_Skills = df[df['job_level'] == levels]['job_skills'].explode()
#     wordcloud = WordCloud(width=700, height=400, background_color="white").generate(' '.join(levels_Skills))
#     plt.figure(figsize=(10, 6))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis('off')
#     plt.title(f"Word Cloud for {levels}")
#     plt.savefig(f"wordcloud_{levels}.png")  # Save plot within the loop
#     plt.show()


## Random Forest Model Implementation
## creating strings from skills column
df["skills_strings"] = df['job_skills'].apply(lambda x: ' '.join(x))
vectorizer = CountVectorizer(max_features=500)
X = vectorizer.fit_transform(df['skills_strings'])
y = df['job_level']

## Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test) 

print("Random Forest Model Summary:")
print(classification_report(y_test, y_pred)) 
## Viualizing results
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix for Random Forest')
plt.savefig("Confusion_Matrix_for_Random_Forest.png")