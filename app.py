import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from easyFSL_helper import *
from dataset_helper import *

@st.cache_resource
def load_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device) 
    return device


@st.cache_resource
def load_data(data):
    if data == "Animals":
        dataset_path = 'animals_dataset'
        labelmap_path = 'animals_label_map.json'
    else:
        dataset_path = 'flowers_dataset'
        labelmap_path = 'flowers_label_map.json'

    N = 20  # n_classes
    K = 5   # n_support_images
    n_query = 1

    val_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = 128
    loss_module = nn.CrossEntropyLoss()

    class_data = ClassData.get_class_data(dataset_path, labelmap_path, val_transform)
    dataset = CustomDataset(class_data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print('ucitava data')
    return N, K, n_query, val_transform, class_data, data_loader


@st.cache_resource
def load_model(model_type, data):
    if model_type == 'Swin_V2 - najbolji za klasifikaciju životinja':
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        model.head = nn.Flatten()
    # elif model_type == 'ResNet50 - najbolji za klasifikaciju biljaka':
    else:
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        flatten_layer = nn.Flatten()
        model = nn.Sequential(
            resnet,
            flatten_layer
        )

    model = model.to(device)
    print('ucitava m')
    return model

@st.cache_resource
def load_support_set(_model, _data_loader, _N, _K, _n_query, model_type, data):
    embeddings_df = predict_embeddings(_data_loader, _model, device=device)
    features_dataset = FeaturesDataset.from_dataframe(embeddings_df)
    task_sampler = TaskSampler(features_dataset, n_way=_N, n_shot=_K, n_query=_n_query, n_tasks=100)
    features_loader = DataLoader(features_dataset, batch_sampler=task_sampler, pin_memory=True, collate_fn=task_sampler.episodic_collate_fn)
    support_images, support_labels, query_images, query_labels, _ = next(iter(features_loader))
    print('ucitava ss', model_type, data)
    return support_images, support_labels

def on_data_change():
    st.session_state.model = None
    st.session_state.classifier = None
    st.session_state.data_loader = None

def on_selectbox_change():
    st.session_state.model = None
    st.session_state.classifier = None
    st.session_state.support_images = None
    st.session_state.support_labels = None

def predict(model, support_images, support_labels, query_images):
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    pred_labels = torch.max(predictions, 1)[1]
    print('vrti pred')
    return pred_labels

def get_query_images(image, model, device):
    images = image.unsqueeze(0)
    if device is not None:    
        images = images.to(device)
    print('vrti query')
    return model(images).detach().cpu()


device = load_device()

st.title("Klasifikacija rijetkih životinja")

if 'data' not in st.session_state:
    st.session_state.data = "Animals"
    st.session_state.model = None
    st.session_state.classifier = None
    st.session_state.N = None
    st.session_state.K = None
    st.session_state.n_query = None
    st.session_state.val_transform = None
    st.session_state.class_data = None
    st.session_state.data_loader = None

data = st.selectbox(
    "Odaberite model za klasifikaciju", 
    ["Animals",
     "Plants"],
    key='data',
    on_change=on_data_change
)

if st.session_state.data_loader is None:
    st.session_state.N, st.session_state.K, st.session_state.n_query, st.session_state.val_transform, st.session_state.class_data, st.session_state.data_loader = load_data(data)


if 'model_type' not in st.session_state:
    st.session_state.model_type = "Swin_V2 - najbolji za klasifikaciju životinja"
    st.session_state.model = None
    st.session_state.classifier = None
    st.session_state.support_images = None
    st.session_state.support_labels = None

model_type = st.selectbox(
    "Odaberite model za klasifikaciju", 
    ["Swin_V2 - najbolji za klasifikaciju životinja",
     "ResNet50 - najbolji za klasifikaciju biljaka",
     "MyFSL"],
    key='model_type',
    on_change=on_selectbox_change
)

if st.session_state.model is None:
    st.session_state.model =  load_model(st.session_state.model_type, data)
    st.session_state.support_images, st.session_state.support_labels = load_support_set(st.session_state.model, st.session_state.data_loader, st.session_state.N, st.session_state.K, st.session_state.n_query, model_type, data)

if st.session_state.classifier is None:
    print('ucitava classifier')
    st.session_state.classifier = PrototypicalNetworks(backbone=nn.Identity())

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if st.button('Process Image'):
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        query_image = st.session_state.val_transform(image)
        query_images = get_query_images(query_image, st.session_state.model, device)
        prediction_label = predict(st.session_state.classifier, st.session_state.support_images, st.session_state.support_labels, query_images)

        st.image(image, caption='Uploaded Image.')
        st.text("Processing the image...")

        result = f"Predpostavljena vrsta je {st.session_state.class_data[prediction_label.item()].class_name}\nNaziv: {st.session_state.class_data[prediction_label.item()].cro_name}\n{st.session_state.class_data[prediction_label.item()].description}"
        
        st.text(result)
    else:
        st.text("Please upload an image before clicking the button.")