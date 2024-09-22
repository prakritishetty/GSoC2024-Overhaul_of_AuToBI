import csv
import os
import datasets

logger = datasets.logging.get_logger(__name__)

_DATA_URL = "./BURSC_DATA_TRIAL.zip"
_ALL_CONFIGS  = []

class DummyConfig(datasets.BuilderConfig):

    def __init__(
        self, name, data_url
    ):
        super(DummyConfig, self).__init__(
            name=self.name,
            version=datasets.Version("1.0.0", ""),

        )
        self.name = name
        self.data_url = data_url

def _build_config(name):
    return DummyConfig(
        name=name,
        data_url=_DATA_URL,
    )

class Dummy(datasets.GeneratorBasedBuilder):

    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [_build_config(name) for name in _ALL_CONFIGS + ["all"]]


    def _info(self):

        features = datasets.Features(
            {
                
                "ID": datasets.Value("string"),
                "name": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "start_time": datasets.Value("string"),
                "end_time": datasets.Value("string"),
                "label": datasets.ClassLabel( names=[0,3,4] )
                
            }
        )

        return datasets.DatasetInfo(
            features=features,
            supervised_keys=("audio", "label"),
        )



    def _split_generators(self, dl_manager):
        #   langs = (
        #     _ALL_CONFIGS
        #     if self.config.name == "all"
        #     else [self.config.name]
        # )
        # print("DATA URL", self.config.data_url)
        archive_path = dl_manager.download_and_extract(self.config.data_url)
        # print("ARCHIVE PATH", archive_path)
        audio_path = dl_manager.extract(
          os.path.join(archive_path, "BURSC_DATA_TRIAL", "audio.zip"))
        # print("AUDIO PATH NEW", audio_path)


        text_path = dl_manager.extract(
            os.path.join(archive_path, "BURSC_DATA_TRIAL", "text.zip"))
        print("TEXT PATHS NEW", text_path)

        # print(os.listdir(text_path))
        # print(os.listdir(os.path.join(text_path, "text")))

        text_path = os.path.join(text_path, "text", "bursc_annotations.csv")
        return [
          datasets.SplitGenerator(
              name=datasets.Split.TRAIN,
              gen_kwargs={
                  "audio_path": audio_path,
                  "text_paths": text_path,
              },
            )
        ]





    def _generate_examples(self, audio_path, text_paths):
          key = 0
          print("AUDIO PATH", audio_path)
          print("TEXT PATHS", text_paths)
         
          with open(text_paths, encoding="utf-8") as csv_file:
              csv_reader = csv.reader(csv_file, delimiter=",", skipinitialspace=True)
              next(csv_reader)
              for row in csv_reader:
                  newid, ID, name, start_time, end_time, embeddings, label = row
                  print("ROW", row)
                  file_path = os.path.join(audio_path, "audio", name)
                  print("FILE PATH", file_path)
                  yield key, {
                    "ID":ID,
                    "name":name,
                    "audio":file_path,
                    "start_time":start_time,
                    "end_time":end_time,
                    "label":label
                }
                  key += 1
