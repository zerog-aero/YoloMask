import json
import cv2
from pathlib import Path, PurePosixPath
import numpy as np
import datetime
import logging
import random
import tqdm
from time import perf_counter

class CocoDataset():
    """
    Creates standard coco stile json format label file from input label format.

    For further information on coco dataset see: http://cocodataset.org/#format-data
    """

    def __init__(self, img_path, label_path, img_suffix="jpg"):
        self.img_suffix = img_suffix
        self.img_path = img_path
        self.label_path = label_path
        self.coco_dataset = dict()
        self.logger = logging.getLogger(__name__)
        self.anno_idx = 0
        self.created_info = False
        self.created_license = False
        self.lists_init = False
    
    def create_info(self, description=None, url=None, version="1.0", year=None, contributor=None, created=None):
        """
        Create info section in json file

        Parameters
        ----------
        description : str or None, optional
            dataset description, by default None
        url : str or None, optional
            info url, by default None
        version : str or None, optional
            dataset version, by default "1.0"
        year : str or None, optional
            dataset year info, will be set to current year if None, by default None
        contributor : str, optional
            contributor info, by default None
        created : str, optional
            creation date, will be set to current date if None, by default None
        """

        now = datetime.datetime.now()
        if year is None:
            year = f'{now:%Y}'
        
        if created is None:
            created = f'{now:%Y-%m-%d}'

        info = {'description': description,
                'url': url,
                'version': version,
                'year': year,
                'contributor': contributor,
                'data_created': created}

        self.coco_dataset["info"] = info
        self.created_info = True
    
    def create_license(self, url=None, id=1, name=None):
        """
        Create license section of label file

        Parameters
        ----------
        url : str, optional
            license url, by default None
        id : int, optional
            license id, by default 1
        name : str, optional
            license name, by default None
        """

        license = [{'url': url,
                    'id': id,
                    'name': name}]
        self.coco_dataset["license"] = license
        self.created_license = True

    def _create_categories(self):
        """
        Create category section of label file.

        Categories will be generated from entity input file. ids are 1-indexed!
        """

        self.coco_dataset["categories"] = list()
        for id_, cat in self.entities_id.items():
            self.coco_dataset["categories"].append({"id": id_, "name": cat})

    def _read_data(self):
        """
        Generate list of images and json files.
        """

        self.jsons_raw = sorted(list(Path(self.label_path).glob("*.json")))
        self.images, self.jsons = self._filter_img_list(self.jsons_raw)

    def _files_split(self, train_test=0.8, mode="train"):
        """
        Split dataset into train and validation

        Parameters
        ----------
        train_test : float, optional
            train set size, by default 0.8
        mode : str {"train", "val"}, optional
            create the train oder val set, by default "train"
        """
        
        if not self.lists_init:
            self.logger.info(f"Splitting Train({train_test:.2f})/Test({1 - train_test:.2f})")
            data_len = list(range(len(self.jsons)))
            train_nbr = int(len(self.jsons) * train_test) 
            self.train_choice = sorted(random.sample(data_len, k = train_nbr))
            self.test_choice = sorted(list(set(data_len).difference(set(self.train_choice))))
            self.images_all, self.jsons_all = self.images, self.jsons
            self.lists_init = True
        if mode=="train":
            self.logger.info(f"Creating Train Dataset")
            self.images = np.array(self.images_all)[self.train_choice].tolist()
            self.jsons = np.array(self.jsons_all)[self.train_choice].tolist()
        if mode=="val":
            self.logger.info(f"Creating Validation Dataset")
            self.images = np.array(self.images_all)[self.test_choice].tolist()
            self.jsons = np.array(self.jsons_all)[self.test_choice].tolist()
    
    def _filter_img_list(self, label_list):
        """
        Refine image and label list.

        Only images with corresponding json file are taken. If json exists without corresponding img, skip 
        file.

        Parameters
        ----------
        label_list : list
            input raw label list

        Returns
        -------
        tuple[list, list]
            list of image files, list of json files
        """

        new_img = list()
        json_lst = list()
        for json in label_list:
            img_path = Path(self.img_path) / json.with_suffix(f".{self.img_suffix}").name
            if img_path.is_file():
                new_img.append(img_path)
                json_lst.append(json)
            else:
                self.logger.warning(f"Image {img_path} not found. Skipping")
        return new_img, json_lst

    def _image_dim(self, path_):
        """
        Get image dimensions.

        Read image with cv2 and get shape

        Parameters
        ----------
        path_ : str
            Path to image file

        Returns
        -------
        tuple[int, int]
            image width dimension in px, image height dimension in px
        """

        im = cv2.imread(path_)
        height, width, _ = im.shape
        return width, height

    def create_image_entry(self, file_name, height, width, id_):
        """
        Create image entry in label file

        Parameters
        ----------
        file_name : str
            image file name
        height : int
            image height
        width : int
            image width
        id_ : int
            image id

        Returns
        -------
        dict
            image entry as dictionary
        """
        img_dict = {"file_name": file_name,
                    "height": height,
                    "width": width,
                    "id": id_}
        return img_dict

    def _read_images(self):
        """
        Read image files and create file entries. 
        """

        self.img_dict = dict()
        self.img_id_dict = dict()
        for i, img in enumerate(self.images):
            w, h = self._image_dim(str(img))
            self.img_dict[img.name] = self.create_image_entry(
            file_name=img.name,
            height=h,
            width=w,
            id_= i)
        self.coco_dataset["images"] = list(self.img_dict.values())

    
    def _load_entity_list(self, input_lst):
        """
        Load entity list and create entity dictionary.

        Label ids are 1-indexed!

        Parameters
        ----------
        input_lst : str
            path to input entity file
        """

        self.entities = dict()
        self.entities_id = dict()
        
        if input_lst is None:
            for i in range(1, 256):
                self.entities[str(i)] = i
                self.entities_id[i] = str(i)
        else:
            with open(input_lst) as inf:
                for i, line in enumerate(inf):
                    id_ = i + 1
                    self.entities[line.strip()] = id_
                    self.entities_id[id_] = line.strip()

    def _read_labels(self, split_label=None):
        """
        Create complete annotation entry in label file 

        Parameters
        ----------
        split_label : list[str, int], optional
            how to split a name label, first entry defines where to split a str
            second entry what part of the split to take, e.g. plane-1 (split at "_" take first part),
            by default None
        """

        self.annotations = list()
        for json in tqdm.tqdm(self.jsons):
            self.annotations.append(self._create_label_entry_labelme(json, iscrowd=0, split_label=split_label))
        self.coco_dataset["annotations"] =  np.concatenate(self.annotations).tolist()

    def _polygon_to_box(self, points):
        """
        Transform to xywh format as in Coco dataset


        box coordinates are measured from the top left image corner and are 0-indexed
        See http://cocodataset.org/#format-data
        """
         
        max_pos = np.max(points, axis=0)
        min_pos = np.min(points, axis=0)
        x = min_pos[0]
        y = min_pos[1]
        width = (max_pos[0] - min_pos[0])
        height = (max_pos[1] - min_pos[1])
        return np.around([x, y, width, height], decimals=3).tolist()


    def _get_image_id(self, data):
        return self.img_dict[data["imagePath"]]["id"]

    def _create_label_entry_labelme(self, path_, iscrowd=0, skip_person=True, split_label=None, area=None):
        """
        Create single annotations entry in label file

        Parameters
        ----------
        path_ : str
            path to json file
        iscrowd : int, optional
            coco crowd option, by default 0
        skip_person : bool, optional
            whether the entry "person" should be skipped, by default True
        split_label : list[str, int], optional
            how to split a name label, first entry defines where to split a str
            second entry what part of the split to take, e.g. plane-1 (split at "_" take first part),
            by default None
        area : float, optional
            polygon area, if none use cv2 contourArea methode to compute area.
            THIS WILL YIELD AN INTEGER AND MIGHT BE INACCURATE , by default None

        Returns
        -------
        list[dict]
            list of entry dictionaries for one label file
        """
        
        label_list = []
        with open(path_) as inf:
            json_data = json.load(inf)
            for entity in json_data["shapes"]:
                if split_label is None:
                    label = entity["label"]
                else:
                    label = entity["label"].split(split_label[0])[split_label[1]] 
                if skip_person and "person" in label:
                    continue
                points = entity["points"]
                bbox = self._polygon_to_box(points)
                area = bbox[2] * bbox[3]
                if area is None:
                    poly_area = cv2.contourArea(np.array(points).astype(np.int))
                else:
                    poly_area = area
                label_dict = {"category_id": self.entities[label], 
                            "iscrowd": iscrowd,
                            "image_id": self._get_image_id(json_data),
                            "id": self.anno_idx,
                            "segmentation": [np.around(np.array(points), decimals=2).flatten().tolist()],
                            "bbox": bbox,
                            "area": poly_area,
                            }
                label_list.append(label_dict)
                self.anno_idx += 1
        return label_list

    def _save(self, outname):
        """
        Save coco stile label file

        Parameters
        ----------
        outname : str
            path to output file
        """

        with open(outname, "w") as write_file:
            json.dump(self.coco_dataset, write_file, indent=4)

    def convert_labelme(self, entity_file, output, split_label=["-", 0], train_split=None):
        """
        Method to convert labelme stile label files. Entry point

        Parameters
        ----------
        entity_file : str
            path to input entity file
        output : str
            path to coco style output file
        split_label : list[str, int], optional
            how to split a name label, first entry defines where to split a str
            second entry what part of the split to take, e.g. plane-1 (split at "_" take first part),
            by default None
        train_split : None or float
            fraction for train test dataset , by default None = 100% in train set
        """
        
        if not self.created_info:
            self.create_info()
        if not self.created_license:
            self.create_license()
        self._load_entity_list(entity_file)
        self._read_data()
        
        tt = perf_counter()
        if train_split is None:
            self._read_images()
            self._read_labels(split_label=split_label)
            self._create_categories()
            self._save(output)
        else:
            for mode_ in ["train", "val"]:
                self._files_split(train_test=train_split, mode=mode_)
                self._read_images()
                self._read_labels(split_label=split_label)
                self._create_categories()
                self._save(f"{mode_}_{output}")
        self.logger.info(f"Time: {perf_counter() - tt:.2f} s")


class Yolov3Dataset(CocoDataset):
    def __init__(self, img_path, label_path, img_suffix="jpg"):
        self.img_suffix = img_suffix
        self.img_path = img_path
        self.label_path = label_path
        self.coco_dataset = dict()
        self.logger = logging.getLogger(__name__)
        self.anno_idx = 0
        self.lists_init = False
        self.img_dims = dict()
    
    def _load_entity_list(self, input_lst):
        """
        Load entity list and create entity dictionary.

        Label ids are 1-indexed!

        Parameters
        ----------
        input_lst : str
            path to input entity file
        """

        self.entities = dict()
        self.entities_id = dict()

        with open(input_lst) as inf:
            for i, line in enumerate(inf):
                id_ = i
                self.entities[line.strip()] = id_
                self.entities_id[id_] = line.strip()

    def convert_to_yolov3(self, entity_file, outfolder, split_label=["-", 0], train_split=None, dataset_path=None, 
                          outname="dataset.txt"):
        """
        Method to convert labelme stile label files. Entry point

        Parameters
        ----------
        entity_file : str
            path to input entity file
        output : str
            path to coco style output file
        split_label : list[str, int], optional
            how to split a name label, first entry defines where to split a str
            second entry what part of the split to take, e.g. plane-1 (split at "_" take first part),
            by default None
        train_split : None or float
            fraction for train test dataset , by default None = 100% in train set
        """
        self._load_entity_list(entity_file)
        self._read_data()
        
        tt = perf_counter()
        if train_split is None:
            self._read_images()

            self._read_labels(split_label=split_label, outfolder=outfolder)
        else:
            for mode_ in ["train", "val"]:
                self._files_split(train_test=train_split, mode=mode_)
                self._read_images()
                self._read_labels(split_label=split_label, outfolder=outfolder)
        self._save_img_paths(out=dataset_path, outname=outname)
        self.logger.info(f"Time: {perf_counter() - tt:.2f} s")

    def _create_label_yolov3(self, path_, outfolder, iscrowd=0, skip_person=True, split_label=None, area=None):
        """
        Create single annotations entry in label file

        Parameters
        ----------
        path_ : str
            path to json file
        iscrowd : int, optional
            coco crowd option, by default 0
        skip_person : bool, optional
            whether the entry "person" should be skipped, by default True
        split_label : list[str, int], optional
            how to split a name label, first entry defines where to split a str
            second entry what part of the split to take, e.g. plane-1 (split at "_" take first part),
            by default None
        area : float, optional
            polygon area, if none use cv2 contourArea methode to compute area.
            THIS WILL YIELD AN INTEGER AND MIGHT BE INACCURATE , by default None

        Returns
        -------
        list[dict]
            list of entry dictionaries for one label file
        """
        label_list = []
        with open(path_) as inf:
            json_data = json.load(inf)
            for entity in json_data["shapes"]:
                if split_label is None:
                    label = entity["label"]
                else:
                    label = entity["label"].split(split_label[0])[split_label[1]] 
                if skip_person and "person" in label:
                    continue
                points = entity["points"]
                w, h = self.img_dims[path_.name]
                bbox = self._polygon_to_box(points, w, h)
                bbox.insert(0, self.entities[label])
                label_list.append(bbox)
        self._save(path_, label_list, outfolder)

    def _read_labels(self, split_label=None, outfolder="test"):
        """
        Create complete annotation entry in label file 

        Parameters
        ----------
        split_label : list[str, int], optional
            how to split a name label, first entry defines where to split a str
            second entry what part of the split to take, e.g. plane-1 (split at "_" take first part),
            by default None
        """
        for json in tqdm.tqdm(self.jsons):
            self._create_label_yolov3(json, outfolder, iscrowd=0, split_label=split_label)

    def _polygon_to_box(self, points, w, h):
        """
        Transform to xywh format as in Coco dataset


        yolov3 coordinates are x_center y_center width height
        """
        max_pos = np.max(points, axis=0)
        min_pos = np.min(points, axis=0)
        x = (max_pos[0] + min_pos[0]) / 2 / w
        y = (max_pos[1] + min_pos[1]) / 2 / h
        width = (max_pos[0] - min_pos[0]) / w
        height = (max_pos[1] - min_pos[1]) / h
        return np.around([x, y, width, height], decimals=3).tolist()

    def _read_images(self):
        """
        Read image files and create file entries. 
        """

        self.img_dict = dict()
        self.img_id_dict = dict()
        for i, img in enumerate(self.images):
            w, h = self._image_dim(str(img))
            self.img_dims[img.with_suffix(".json").name] = [w, h]

    def _save(self, outname, data, outfolder):
        """
        Save coco stile label file

        Parameters
        ----------
        outname : str
            path to output file
        """
        name = str(Path(outfolder) / outname.with_suffix(".txt").name)
        with open(name, "w") as write_file:
            for i, el in enumerate(data):
                write_file.write(" ".join(map(str, el)))
                if i < (len(data) - 1):
                    write_file.write("\n")

    def _save_img_paths(self, out=None, outname="dataset.txt"):
        if out is None:
            out_ = str(Path(self.img_path).parent/"dataset.txt")
        else:
            out_ = str(Path(out)/outname)
        with open(out_, "w") as outf:
            for img in self.images:
                outf.write(f"{PurePosixPath(img)}\n")

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)7s (%(asctime)s) (%(name)s/%(funcName)s): %(message)s', level=logging.INFO)
    
    IMG_PATH = "data/face_mask_2"
    ds_train = Yolov3Dataset(img_path=f"{IMG_PATH}/images/train", 
                            label_path=f"{IMG_PATH}/labels/train_json",
                            img_suffix="jpg")

    ds_train.convert_to_yolov3(entity_file="data/face_mask/fm.names", outfolder=f"{IMG_PATH}/labels/train",
                               dataset_path=IMG_PATH, outname="train.txt")

    ds_val = Yolov3Dataset(img_path=f"{IMG_PATH}/images/val", 
                            label_path=f"{IMG_PATH}/labels/val_json",
                            img_suffix="jpg")

    ds_val.convert_to_yolov3(entity_file="data/face_mask/fm.names", outfolder=f"{IMG_PATH}/labels/val",
                             dataset_path=IMG_PATH, outname="val.txt")
