# VNPT DVC SearchAPI

## I. Use

Run `DVC_SearchAssistant_V1.exe`

[http://localhost:5002/?input_text=Giấy tờ cần thiết để mình khởi nghiệp?](http://localhost:5002/?input_text=Giấy%20tờ%20cần%20thiết%20để%20mình%20khởi%20nghiệp?)

## II. Dev

### 1. Install
Run `install.bat`

### 2a. Gradio App
```
gradio app.py
```

### 2b. REST API
```
py api.py
```

### 3. Build API binary
```
py _build.py
```