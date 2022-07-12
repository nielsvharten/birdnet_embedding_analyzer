from logging import exception
import urllib.request, json
from importlib_metadata import metadata
import pandas as pd 
from pip._vendor import requests
import pydub
import re
from os.path import exists

# retrieve all recordings for a given query (when leaving page, num_pages and recordings unspecified)
# stored in recs.txt
def fetch_pages(query, page=1, num_pages=1, recordings=None):
    if recordings == None:
        recordings = open("recs.txt", "w+")
        recordings.write("id\tname\n")

    url = "https://xeno-canto.org/api/2/recordings?query=" + query + "&page="  + str(page)
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        num_pages = int(data['numPages'])

        for i in range(len(data['recordings'])):
            recordings.write(data['recordings'][i]['id'] + "\t" + data['recordings'][i]['en'] + "\n")

    if page < num_pages:
        fetch_pages(query, page+1, num_pages, recordings)
    else:
        recordings.close()


# filter recordings from recs.txt, store in filtered_recs.txt
# keep only birds with >= 20 samples and throw out soundscapes and unknown identities
def filter_recordings():
    recs = pd.read_csv("recs.txt", sep="\t", encoding="latin-1")
    
    # Keep only classes with at least 20 audio samples 
    recs_per_bird = recs.groupby(['name']).size().reset_index(name='counts')
    birds_to_keep = recs_per_bird[recs_per_bird["counts"] >= 20]
    filtered_recs = recs[recs["name"].isin(birds_to_keep['name'])]

    # Keep only classes with bird name
    filtered_recs = filtered_recs[~filtered_recs["name"].isin(["Identity unknown", "Soundscape"])]

    # Rename birds to match BirdNET namings
    filtered_recs = rename_birds(filtered_recs)

    filtered_recs.to_csv("filtered_recs.txt", index=None, sep='\t')
    print("Created filtered_recs.txt")


# download all mp3-files for filtered_recs.txt
# convert to wav-files
def download_mp3s():
    recordings = pd.read_csv("filtered_recs.txt", sep="\t")

    with requests.Session() as req:
        for i, row in recordings.iterrows():
            id = row['id']

            if exists("mp3_files\\" + str(id) + ".mp3"):
                continue

            url = "https://xeno-canto.org/" + str(id) + "/download"
            download = req.get(url)

            if download.status_code == 200:
                with open("mp3_files/" + str(id) + ".mp3", 'wb') as f:
                    f.write(download.content)
                
                try:
                    sound = pydub.AudioSegment.from_mp3("./mp3_files/" + str(id) + ".mp3")
                    sound.export("./wav_files/" + str(id) + ".wav", format="wav")
                except:
                    print("WAV conversion gave error for file: " + str(id))
            else:
                print(f"Download Failed For File {id}")


def rename_birds(dataframe):
    bird_dict = {
        "Common Blackbird": "Eurasian Blackbird",
        "Common Grasshopper Warbler": "Common Grasshopper-Warbler",
        "Common Linnet": "Eurasian Linnet",
        "Common Moorhen": "Eurasian Moorhen",
        "Common Reed Bunting": "Reed Bunting",
        "Common Starling": "European Starling",
        "Common Whitethroat": "Greater Whitethroat",
        "European Green Woodpecker": "Eurasian Green Woodpecker",
        "European Nightjar": "Eurasian Nightjar",
        "Grey Heron": "Gray Heron",
        "Greylag Goose": "Graylag Goose",
        "Northern Raven": "Common Raven",
        # "River Warbler": "",
        "Western Jackdaw": "Eurasian Jackdaw",
        "Woodlark": "Wood Lark"
    }

    for bird in bird_dict:
        print(bird)
        print(bird_dict[bird])
        dataframe['name'] = dataframe['name'].replace(bird, bird_dict[bird])
    
    return dataframe


fetch_pages("cnt:netherlands+q:A")
# filter_recordings()
# download_mp3s()
