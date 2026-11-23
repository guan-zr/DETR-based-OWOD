"""Train images never load XML. Offline evaluation is a separate API."""
from pathlib import Path
from functools import lru_cache
import json
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF
from datasets.transforms import resize as detector_resize

KNOWN = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
         'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')
ALIASES = dict(zip(('airplane','dining table','motorcycle','potted plant','couch','tv'),
                   ('aeroplane','diningtable','motorbike','pottedplant','sofa','tvmonitor')))


class ImageOnlyDataset(Dataset):
    def __init__(self, root, split='owod_t1_train', short_side=800, max_size=1333, unique=False):
        self.root = Path(root)
        self.ids = (self.root/'ImageSets/TOWOD'/f'{split}.txt').read_text().splitlines()
        if unique:
            self.ids = list(dict.fromkeys(self.ids))
        self.short_side, self.max_size = short_side, max_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stem = self.ids[index]
        with Image.open(self.root/'JPEGImages'/f'{stem}.jpg') as image:
            image = image.convert('RGB')
            original_hw = (image.height, image.width)
            # Same deterministic resize/normalization as detector test preprocessing.
            # Reuse official max_size rounding (generic TorchVision resize can
            # differ by one pixel on elongated images). No target/XML is needed.
            image, _ = detector_resize(image, None, self.short_side, max_size=self.max_size)
            resized_hw = (image.height, image.width)
            tensor = TF.normalize(TF.to_tensor(image), [0.485,0.456,0.406], [0.229,0.224,0.225])
        return {'image':tensor, 'image_id':stem, 'original_hw':original_hw, 'resized_hw':resized_hw}


def image_collate(items):
    return items


@lru_cache(maxsize=1)
def evaluation_vocabulary():
    path=Path(__file__).resolve().parent/'configs/mowodb_task1_classes.json'
    names=json.loads(path.read_text())
    return set(names['known']+names['unknown'])


def read_eval_gt(root, stem):
    tree = ET.parse(Path(root)/'Annotations'/f'{stem}.xml').getroot()
    boxes, known, difficult, names = [], [], [], []
    for obj in tree.findall('object'):
        name = obj.findtext('name'); name = ALIASES.get(name, name)
        if name not in evaluation_vocabulary():
            raise ValueError(f'Class outside audited M-OWODB 80-class vocabulary: {name!r}')
        box = [float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
        box[0] -= 1; box[1] -= 1
        boxes.append(box); known.append(name in KNOWN); names.append(name)
        difficult.append(bool(int(obj.findtext('difficult','0'))))
    return {'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            'known':torch.tensor(known,dtype=torch.bool),
            'difficult':torch.tensor(difficult,dtype=torch.bool),'names':names}
