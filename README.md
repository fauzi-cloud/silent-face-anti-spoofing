face_anti_spoofing/
│
├── data/ # Dataset Location
│ ├── fake/ # Fake Dataset Location
│ └── real/ # Real Dataset Location
│
├── raw/ # Dir to Save Raw Data Before Extract Frame
│
├── models/
│ └── model.py
|
├── result/ # Dir for Save Result Data Train
│
├── utils/
│ └── visualizer.py
│
├── realtime_detection/
│
├── collect_data/
|
├── extract_frame/
|
├── live_detection/
|
├── train_detection_auto/
│
├── requirements.txt
└── README.md

Run This Code First :
python -m venv .venv ( install virtual environment )
.venv\Script\activate ( run virtual environment )
pip install -r requirements.txt ( install requirements )
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ( install torch cpu version )

Extract Frame :
python extract_frame.py

Collect Data :
python collect_data.py

Train Model :
python train.py

Test Live Detection :
python live_detection.py

Train Detection :
python train_detection.py
