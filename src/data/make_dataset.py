import os
import pathlib
import zipfile

def download_kaggle_data():
    base_dir = pathlib.Path(__file__).resolve().parents[2]
    kaggle_config_dir = base_dir / "secrets"
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)

    # Importer ici, après avoir défini la variable d'environnement
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_dir = base_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Authenticating Kaggle API using config dir: {kaggle_config_dir}")
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.competition_download_files(
        competition="house-prices-advanced-regression-techniques",
        path=str(data_dir)
    )

    zip_path = data_dir / "house-prices-advanced-regression-techniques.zip"
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print("Cleaning up...")
    zip_path.unlink()

    print(f"✅ Dataset downloaded and extracted to {data_dir}")

if __name__ == "__main__":
    download_kaggle_data()
