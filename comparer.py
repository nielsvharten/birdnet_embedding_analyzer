from matplotlib.lines import Line2D
from more_itertools import first
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

from matplotlib import colors
from scipy import spatial
import librosa
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
import umap
import matplotlib as mpl


# load embeddings
embeddings = pd.read_csv('embeddings.txt', sep="\t", header=None)
embeddings.columns = ['species', 'id', 'start_time', 'embedding']
embeddings['embedding'] = embeddings['embedding'].apply(lambda s: np.array(list(map(float, s.split(",")))))

recs_per_class = embeddings.groupby(['species'])['species'].count()
min_recs = min(recs_per_class)
max_recs = max(recs_per_class)
avg_recs = sum(recs_per_class) / len(recs_per_class)

# get mean embeddings
mean_embeddings = embeddings.groupby('species')['embedding'].apply(np.mean)


# average distance to mean for a single bird: how varied is the data for each species?
def get_avg_dist_mean():
    avg_dist_means = {}
    for bird in mean_embeddings.index:
        mean_embedding = mean_embeddings[bird]
        all_embeddings = embeddings[embeddings.species == bird]['embedding'].tolist()
        
        avg_dist_mean_bird = [spatial.distance.cosine(mean_embedding, x) for x in all_embeddings] 

        avg_dist_means[bird] = round(sum(avg_dist_mean_bird) / len(avg_dist_mean_bird) * 100, 2)

    sorted_std_devs = {k: v for k, v in sorted(avg_dist_means.items(), key=lambda item: item[1])}
    print(list(sorted_std_devs.items())[:10])
    print(list(sorted_std_devs.items())[-10:])
    
    return avg_dist_means


# which combi of birds are the most similar, and which most different?
def get_most_least_similar_pairs():
    pairs = {}
    for bird1 in mean_embeddings.index:
        for bird2 in mean_embeddings.index:
            # do not compare bird with itself
            if bird1 != bird2:
                distance = spatial.distance.cosine(mean_embeddings[bird1], mean_embeddings[bird2])
                if bird1 < bird2:
                    pairs[bird1 + " vs " + bird2] = distance
                else:
                    pairs[bird2 + " vs " + bird1] = distance

    sorted_pairs = {k: v for k, v in sorted(pairs.items(), key=lambda item: item[1])}
    print(list(sorted_pairs.items())[:10])
    print(list(sorted_pairs.items())[-10:])


def get_distance_pair(bird1, bird2):
    dist = spatial.distance.cosine(mean_embeddings[bird1], mean_embeddings[bird2])
    
    return dist


'''
def visualize_birds_mean():
    features = mean_embeddings.to_list()
    std_features = StandardScaler().fit_transform(features)

    standard_embedding = umap.UMAP(random_state=42).fit_transform(std_features)
    classes = LabelEncoder().fit(mean_embeddings.index).transform(mean_embeddings.index)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=classes, s=1, cmap='Spectral')

    # annotate all point with bird names
    for i, txt in enumerate(mean_embeddings.index):
        plt.annotate(txt, (standard_embedding[:, 0][i], standard_embedding[:, 1][i]))
    
    plt.show()


def visualize_all_birds():

    birds =  mean_embeddings.index.to_list()
    print(birds)
    # make clustering for a bird, how many types of calls
    features = embeddings['embedding'].to_list() + mean_embeddings.to_list()
    #std_features = StandardScaler().fit_transform(features)

    standard_embedding = umap.UMAP(random_state=42, min_dist=0.0, n_neighbors=200, metric="cosine").fit_transform(features)
    labels = embeddings['species'].to_list() + mean_embeddings.index.to_list()
    classes = LabelEncoder().fit(labels).transform(labels)
    
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, max(classes))
    # define the bins and normalize
    N = max(classes)
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.scatter(standard_embedding[:-(N+1), 0], standard_embedding[:-(N+1), 1], c=classes[:-(N+1)], s=5, cmap=cmap, norm=norm)

    birds =  mean_embeddings.index.to_list()
    mean_standard_embedding = standard_embedding[-N:]

    # add means
    plt.scatter(mean_standard_embedding[:, 0], mean_standard_embedding[:, 1], c=classes[-N:], s=15, cmap=cmap, norm=norm)


    for i, row in embeddings.iterrows():
        if row['species'] == "Little Grebe" or row['species'] == "Lesser Spotted Woodpecker":
            plt.annotate(row['species'], (standard_embedding[i, 0], standard_embedding[i, 1]))

    for i_mean in range(N):
        plt.annotate(birds[i_mean], (mean_standard_embedding[i_mean, 0], mean_standard_embedding[i_mean, 1]))

    plt.show()
'''


def visualize_mean_embeddings():
    features = mean_embeddings.to_list()
    std_features = StandardScaler().fit_transform(features)

    standard_embedding = umap.UMAP(random_state=42, min_dist=0.1, n_neighbors=15, metric="cosine").fit_transform(std_features)
    
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=10)

    birds =  mean_embeddings.index.to_list()

    for i in range(len(birds)):
        plt.annotate(birds[i], (standard_embedding[:, 0][i], standard_embedding[:, 1][i]))

    plt.show()


# CURRENTLY ONLY SUITED FOR TWO SPECIES
def visualize_birds(birds):
    bird_embeddings = embeddings[embeddings.species.isin(birds)]
    features = bird_embeddings['embedding'].to_list()
    std_features = StandardScaler().fit_transform(features)

    standard_embedding = umap.UMAP(random_state=42, min_dist=0.0, metric="cosine").fit_transform(std_features)
    cmap = colors.ListedColormap(['blue', 'red'])
    classes = LabelEncoder().fit(bird_embeddings['species']).transform(bird_embeddings['species']) #[embeddings.species.isin(birds)]
    
    fig, ax = plt.subplots()

    ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=classes, s=5, cmap=cmap)
    
    first_species = bird_embeddings['species'].iloc[0]
    first_class = classes[0]

    blue_label = birds[1]
    red_label = birds[0]
    if first_species == birds[0] and first_class == 0 or first_species == birds[1] and first_class == 1:
        blue_label = birds[0]
        red_label = birds[1]

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=blue_label, markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label=red_label, markerfacecolor='red', markersize=10)
    ]
    ax.legend(handles=legend_elements)
    for i in range(len(bird_embeddings)):
        plt.annotate(str(bird_embeddings.iloc[i]['species']) + "_" + str(bird_embeddings.iloc[i]['start_time']), (standard_embedding[:, 0][i], standard_embedding[:, 1][i]))

    plt.show()


def show_spectrogram(id, start_time=0):
    sig, rate  = librosa.load("wav_files\\" + str(id) + ".wav", sr=48000, duration=3, offset=start_time)
    spec = librosa.feature.melspectrogram(sig, rate, fmax=15000)
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(spec, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=rate, fmax=15000, ax=ax)

    plt.savefig("out.png")
    plt.show()


'''
# for each segment calculate distance to closest 
def classify_segments():
    confusion_matrix = {}
    for bird in mean_embeddings.index:
        confusion_matrix[bird] = {}

    for i, row in embeddings.iterrows():
        min_distance = 999999
        prediction = None
        
        # loop through bird means, save closest
        for bird in mean_embeddings.index:
            mean_embedding = mean_embeddings[bird]
            bird_distance = spatial.distance.cosine(row.embedding, mean_embedding)
            if bird_distance < min_distance:
                min_distance = bird_distance
                prediction = bird
        
        # add prediction to confusion matrix
        if prediction in confusion_matrix[row.species]:
             confusion_matrix[row.species][prediction] += 1
        else:
            confusion_matrix[row.species][prediction] = 1

    print(confusion_matrix)
'''


def get_classification_acc():
    nr_correct = 0

    for i, row in embeddings.iterrows():
        min_distance = 999999
        prediction = None
        
        # loop through bird means, save closest
        for bird in mean_embeddings.index:
            mean_embedding = mean_embeddings[bird]
            bird_distance = spatial.distance.cosine(row.embedding, mean_embedding)
            if bird_distance < min_distance:
                min_distance = bird_distance
                prediction = bird
        
        # update accuracy
        if prediction == row.species:
            nr_correct += 1

    return nr_correct / len(embeddings)


def get_classification_acc_3d():
    features = embeddings['embedding'].to_list()
    features += mean_embeddings.to_list()
    std_features = StandardScaler().fit_transform(features)

    #classes = LabelEncoder().fit(embeddings['species']).transform(embeddings['species'])
    all_embeddings_3d = umap.UMAP(random_state=42, n_components=3, min_dist=0.0, n_neighbors=100, metric="cosine").fit_transform(std_features)
    nr_classes = len(mean_embeddings)
    nr_embeddings = len(embeddings)

    # embeddings to classify in 3d
    correct_species = embeddings['species']
    embeddings_3d = all_embeddings_3d[:nr_embeddings]
    print(len(embeddings_3d))
    print(len(correct_species))

    # mean embeddings in 3d
    birds = mean_embeddings.index.to_list()
    mean_embeddings_3d = all_embeddings_3d[-nr_classes:]

    nr_correct = 0
    for i in tqdm(range(nr_embeddings)):
        min_distance = 999999
        prediction = None
        
        # loop through bird means, save closest
        for c in range(nr_classes):
            mean_embedding = mean_embeddings_3d[c]
            bird_distance = spatial.distance.cosine(embeddings_3d[i], mean_embedding)
            if bird_distance < min_distance:
                min_distance = bird_distance
                prediction = birds[c]
        
        # update accuracy
        if prediction == correct_species[i]:
            nr_correct += 1

    return nr_correct / nr_embeddings


def get_classification_acc_2d():
    features = embeddings['embedding'].to_list()
    features += mean_embeddings.to_list()
    std_features = StandardScaler().fit_transform(features)

    #classes = LabelEncoder().fit(embeddings['species']).transform(embeddings['species'])
    all_embeddings_2d = umap.UMAP(random_state=42, min_dist=0.0, n_neighbors=200, metric="cosine").fit_transform(std_features)
    nr_classes = len(mean_embeddings)
    nr_embeddings = len(embeddings)

    # embeddings to classify in 2d
    correct_species = embeddings['species']
    embeddings_2d = all_embeddings_2d[:nr_embeddings]
    print(len(embeddings_2d))
    print(len(correct_species))

    # mean embeddings in 2d
    birds = mean_embeddings.index.to_list()
    mean_embeddings_2d = all_embeddings_2d[-nr_classes:]

    nr_correct = 0
    for i in tqdm(range(nr_embeddings)):
        min_distance = 999999
        prediction = None
        
        # loop through bird means, save closest
        for c in range(nr_classes):
            mean_embedding = mean_embeddings_2d[c]
            bird_distance = spatial.distance.cosine(embeddings_2d[i], mean_embedding)
            if bird_distance < min_distance:
                min_distance = bird_distance
                prediction = birds[c]
        
        # update accuracy
        if prediction == correct_species[i]:
            nr_correct += 1

    return nr_correct / nr_embeddings
