import os
import shutil
import urllib.request
import zipfile

# Determine project directory
NLTK_DATA_DIR = os.path.join(os.getcwd(), 'nltk_data')

# Clear previous partial downloads or locks
shutil.rmtree(NLTK_DATA_DIR, ignore_errors=True)
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

downloads = {
    'corpora/stopwords.zip': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip',
    'corpora/wordnet.zip': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip',
    'corpora/omw-1.4.zip': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip',
    'sentiment/vader_lexicon.zip': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip'
}

def download_and_extract():
    for rel_path, url in downloads.items():
        dir_name = os.path.dirname(rel_path) # e.g. corpora or sentiment
        target_dir = os.path.join(NLTK_DATA_DIR, dir_name)
        os.makedirs(target_dir, exist_ok=True)
            
        zip_path = os.path.join(NLTK_DATA_DIR, rel_path)
        print(f"Downloading {url}...")
        try:
            # Download file
            urllib.request.urlretrieve(url, zip_path)
            print(f"Extracting {rel_path} to {target_dir}...")
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("Cleaning up zip...")
            os.remove(zip_path)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            
    print("NLTK assets downloaded and configured successfully.")

if __name__ == "__main__":
    download_and_extract()
