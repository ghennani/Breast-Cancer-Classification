import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from streamlit_option_menu import option_menu
import time
import pandas as pd
# Set page configuration
st.set_page_config(page_title="Cancer Classification Dashboard", layout="wide")
# Language selection
language = st.sidebar.selectbox("Select Language / Sélectionnez la langue", ["English", "Français"])
# Texts in English and French
texts = {
    "menu": {
        "Home": {"English": "Home", "Français": "Accueil"},
        "Diagnosis": {"English": "Diagnosis", "Français": "Diagnostic"},
        "Treatment": {"English": "Treatment", "Français": "Traitement"},
        "Statistics": {"English": "Statistics", "Français": "Statistiques"}
    },
    "titles": {
        "about_institute": {"English": "About Our Institute", "Français": "À propos de notre institut"},
        "diagnosis": {"English": "Artificial intelligence can detect breast cancer missed by doctors", "Français": "L'intelligence artificielle peut détecter le cancer du sein manqué par les médecins"},
        "treatment_stages": {"English": "Treatment Stages", "Français": "Étapes de traitement"},
        "Statistics_title": {"English": "Global Health Statistics from the World Health Organization (WHO)", "Français": "Statistiques mondiales de la santé de l'Organisation mondiale de la santé (OMS)"}
    },
    "titles_treatment": {
        "BR_surgery":{"English": "Breast cancer surgery: ","Français": "Chirurgie du cancer du sein"},
        "Rad_therapy": {"English": "Radiation therapy", "Français": "Radiothérapie"},
        "Che_therapy": {"English": "Chemotherapy therapy", "Français": "Thérapie de chimiothérapie"},
        "Hormone_therapy": {"English": "Hormone therapy", "Français": "Thérapie hormonale"},
        "Targeted_therapy": {"English": "Targeted therapy", "Français": "Thérapie ciblée"},
        "Immunotherapy": {"English": "Immunotherapy", "Français": "Immunothérapie"},
        "Palliative_care": {"English": "Palliative care", "Français": "Soins palliatifs"}
        },
    "quotes": {
        "English": [
            
            "Every day is a chance to be better."  
        ],
        "Français": [
            
            "Chaque jour est une chance de s'améliorer."
        ]
    },
    "texts_treatment": {
        "BR_surgery": {
            "English": [
                "Breast cancer treatment often begins with surgery to remove the cancer, followed by additional treatments like radiation, chemotherapy, and hormone therapy. Sometimes, chemotherapy or hormone therapy is administered before surgery to shrink the cancer for easier removal. Surgery usually involves removing the cancer and some nearby lymph nodes.",
                "Breast cancer surgery typically involves a procedure to remove the breast cancer and a procedure to remove some nearby lymph nodes.",
                "Complications of breast cancer surgery depend on the procedures you choose. All operations have a risk of pain, bleeding and infection. Removing lymph nodes in the armpit carries a risk of arm swelling, called lymphedema. You may choose to have breast reconstruction after mastectomy surgery. Breast reconstruction is surgery to restore shape to the breast."
            ],
            "Français": [
                "Le traitement du cancer du sein commence souvent par une chirurgie pour enlever le cancer, suivie de traitements supplémentaires comme la radiothérapie, la chimiothérapie et la thérapie hormonale. Parfois, la chimiothérapie ou la thérapie hormonale est administrée avant la chirurgie pour réduire le cancer et faciliter son retrait. La chirurgie consiste généralement à enlever le cancer et quelques ganglions lymphatiques voisins.",
                "La chirurgie du cancer du sein implique généralement une procédure pour enlever le cancer du sein et une procédure pour enlever certains ganglions lymphatiques voisins.",
                "Les complications de la chirurgie du cancer du sein dépendent des procédures choisies. Toutes les opérations comportent un risque de douleur, de saignement et d'infection. L'ablation des ganglions lymphatiques sous le bras comporte un risque de gonflement du bras, appelé lymphœdème. Vous pouvez choisir de reconstruire le sein après une mastectomie. La reconstruction mammaire est une chirurgie pour restaurer la forme du sein."
            ]
        },
        "Rad_therapy": {
            "English": [
                "Radiation therapy treats cancer with powerful energy beams. The energy can come from X-rays, protons or other sources.",
                "For breast cancer treatment, the radiation is often external beam radiation. During this type of radiation therapy, you lie on a table while a machine moves around you. The machine directs radiation to precise points on your body. Less often, the radiation can be placed inside the body. This type of radiation is called brachytherapy.",
                "Radiation therapy is often used after surgery. It can kill any cancer cells that might be left after surgery. The radiation lowers the risk of the cancer coming back.",
                "Side effects of radiation therapy include feeling very tired and having a sunburn-like rash where the radiation is aimed. Breast tissue also may look swollen or feel more firm. Rarely, more-serious problems can happen. These include damage to the heart or lungs. Very rarely, a new cancer can grow in the treated area."
            ],
            "Français": [
                "La radiothérapie traite le cancer avec des faisceaux d'énergie puissants. L'énergie peut provenir de rayons X, de protons ou d'autres sources.",
                "Pour le traitement du cancer du sein, la radiothérapie est souvent une radiothérapie externe. Pendant ce type de radiothérapie, vous vous allongez sur une table pendant qu'une machine se déplace autour de vous. La machine dirige le rayonnement vers des points précis de votre corps. Moins souvent, le rayonnement peut être placé à l'intérieur du corps. Ce type de radiothérapie est appelé curiethérapie.",
                "La radiothérapie est souvent utilisée après une chirurgie. Elle peut tuer toutes les cellules cancéreuses qui pourraient rester après la chirurgie. La radiothérapie réduit le risque de récidive du cancer.",
                "Les effets secondaires de la radiothérapie incluent une grande fatigue et une éruption cutanée semblable à un coup de soleil là où le rayonnement est dirigé. Les tissus mammaires peuvent également sembler enflés ou plus fermes. Rarement, des problèmes plus graves peuvent survenir, comme des lésions cardiaques ou pulmonaires. Très rarement, un nouveau cancer peut se développer dans la zone traitée."
            ]
        },
        "Che_therapy": {
            "English": [
                "Chemotherapy treats cancer with strong medicines. Many chemotherapy medicines exist. Treatment often involves a combination of chemotherapy medicines. Most are given through a vein. Some are available in pill form.",
                "Chemotherapy for breast cancer is often used after surgery. It can kill any cancer cells that might remain and lower the risk of the cancer coming back.",
                "When the cancer spreads to other parts of the body, chemotherapy can help control it. Chemotherapy may relieve symptoms of an advanced cancer, such as pain."
            ],
            "Français": [
                "La chimiothérapie traite le cancer avec des médicaments puissants. De nombreux médicaments de chimiothérapie existent. Le traitement implique souvent une combinaison de médicaments de chimiothérapie. La plupart sont administrés par voie intraveineuse. Certains sont disponibles sous forme de pilules.",
                "La chimiothérapie pour le cancer du sein est souvent utilisée après la chirurgie. Elle peut tuer toutes les cellules cancéreuses qui pourraient rester et réduire le risque de récidive du cancer.",
                "Lorsque le cancer se propage à d'autres parties du corps, la chimiothérapie peut aider à le contrôler. La chimiothérapie peut soulager les symptômes d'un cancer avancé, comme la douleur."
            ]
        },
        "Hormone_therapy": {
            "English": [
                "Hormone therapy uses medicines to block certain hormones in the body. It's a treatment for breast cancers that are sensitive to the hormones estrogen and progesterone. Healthcare professionals call these cancers estrogen receptor positive and progesterone receptor positive. Cancers that are sensitive to hormones use the hormones as fuel for their growth. Blocking the hormones can cause the cancer cells to shrink or die.",
                "Treatments that can be used in hormone therapy include:",
                "Medicines that block hormones from attaching to cancer cells. These medicines are called selective estrogen receptor modulators.",
                "Medicines that stop the body from making estrogen after menopause. These medicines are called aromatase inhibitors.",
                "Surgery or medicines to stop the ovaries from making hormones."
            ],
            "Français": [
                "La thérapie hormonale utilise des médicaments pour bloquer certaines hormones dans le corps. C'est un traitement pour les cancers du sein qui sont sensibles aux hormones œstrogène et progestérone. Les professionnels de la santé appellent ces cancers récepteurs d'œstrogènes positifs et récepteurs de progestérone positifs. Les cancers qui sont sensibles aux hormones utilisent les hormones comme carburant pour leur croissance. Bloquer les hormones peut provoquer la diminution ou la mort des cellules cancéreuses.",
                "Les traitements pouvant être utilisés dans la thérapie hormonale incluent :",
                "Des médicaments qui bloquent les hormones de se fixer aux cellules cancéreuses. Ces médicaments sont appelés modulateurs sélectifs des récepteurs d'œstrogènes.",
                "Des médicaments qui empêchent le corps de produire des œstrogènes après la ménopause. Ces médicaments sont appelés inhibiteurs de l'aromatase.",
                "Une chirurgie ou des médicaments pour empêcher les ovaires de produire des hormones."
            ]
        },
        "Targeted_therapy": {
            "English": [
                "Targeted therapy uses medicines that attack specific chemicals in the cancer cells. By blocking these chemicals, targeted treatments can cause cancer cells to die.",
                "The most common targeted therapy medicines for breast cancer target the protein HER2. Some breast cancer cells make extra HER2. This protein helps the cancer cells grow and survive. Targeted therapy medicine attacks the cells that are making extra HER2 and doesn't hurt healthy cells.",
                "Targeted therapy medicines can be used before surgery to shrink a breast cancer and make it easier to remove. Some are used after surgery to lower the risk that the cancer will come back. Others are used only when the cancer has spread to other parts of the body."
            ],
            "Français": [
                "La thérapie ciblée utilise des médicaments qui attaquent des substances chimiques spécifiques dans les cellules cancéreuses. En bloquant ces substances chimiques, les traitements ciblés peuvent provoquer la mort des cellules cancéreuses.",
                "Les médicaments de thérapie ciblée les plus courants pour le cancer du sein ciblent la protéine HER2. Certaines cellules cancéreuses du sein produisent une quantité excessive de HER2. Cette protéine aide les cellules cancéreuses à croître et à survivre. Les médicaments de thérapie ciblée attaquent les cellules qui produisent cette quantité excessive de HER2 sans nuire aux cellules saines.",
                "Les médicaments de thérapie ciblée peuvent être utilisés avant la chirurgie pour réduire un cancer du sein et faciliter son retrait. Certains sont utilisés après la chirurgie pour réduire le risque de récidive du cancer. D'autres sont utilisés uniquement lorsque le cancer s'est propagé à d'autres parties du corps."
            ]
        },
        "Immunotherapy": {
            "English": [
                "Immunotherapy is a treatment with medicine that helps the body's immune system to kill cancer cells. The immune system fights off diseases by attacking germs and other cells that shouldn't be in the body. Cancer cells survive by hiding from the immune system. Immunotherapy helps the immune system cells find and kill the cancer cells.",
                "Immunotherapy might be an option for treating triple-negative breast cancer. Triple-negative breast cancer means that the cancer cells don't have receptors for estrogen, progesterone or HER2."
            ],
            "Français": [
                "L'immunothérapie est un traitement médicamenteux qui aide le système immunitaire du corps à tuer les cellules cancéreuses. Le système immunitaire combat les maladies en attaquant les germes et autres cellules qui ne devraient pas être dans le corps. Les cellules cancéreuses survivent en se cachant du système immunitaire. L'immunothérapie aide les cellules du système immunitaire à trouver et tuer les cellules cancéreuses.",
                "L'immunothérapie pourrait être une option pour traiter le cancer du sein triple négatif. Le cancer du sein triple négatif signifie que les cellules cancéreuses n'ont pas de récepteurs pour l'œstrogène, la progestérone ou HER2."
            ]
        },
        "Palliative_care": {
            "English": [
                "Palliative care is a special type of healthcare that helps you feel better when you have a serious illness. If you have cancer, palliative care can help relieve pain and other symptoms. A team of healthcare professionals provides palliative care. The team can include doctors, nurses and other specially trained professionals. Their goal is to improve quality of life for you and your family.",
                "Palliative care specialists work with you, your family and your care team to help you feel better. They provide an extra layer of support while you have cancer treatment. You can have palliative care at the same time as strong cancer treatments, such as surgery, chemotherapy or radiation therapy."
            ],
            "Français": [
                "Les soins palliatifs sont un type spécial de soins de santé qui vous aide à vous sentir mieux lorsque vous avez une maladie grave. Si vous avez un cancer, les soins palliatifs peuvent aider à soulager la douleur et d'autres symptômes. Une équipe de professionnels de la santé fournit des soins palliatifs. L'équipe peut inclure des médecins, des infirmières et d'autres professionnels spécialement formés. Leur objectif est d'améliorer la qualité de vie pour vous et votre famille.",
                "Les spécialistes des soins palliatifs travaillent avec vous, votre famille et votre équipe de soins pour vous aider à vous sentir mieux. Ils fournissent une couche supplémentaire de soutien pendant que vous recevez un traitement contre le cancer. Vous pouvez recevoir des soins palliatifs en même temps que des traitements anticancéreux puissants, tels que la chirurgie, la chimiothérapie ou la radiothérapie."
            ]
        }
    },
    "content": {
        "institute_info": {
            "English": "<h4>Breast Cancer Technology Institute is recognised as a world leader and has been featured in:</h4>",
            "Français": "<h4>Institut de Technologie du Cancer du Sein est reconnu comme un leader mondial et a été présenté dans :</h4>"
        },
        "research_shows": {
            "English": "<h3>Our research shows that AI can detect breast cancer with the accuracy of a pathologist</h3>",
            "Français": "<h3>Nos recherches montrent que l'IA peut détecter le cancer du sein avec la précision d'un pathologiste</h3>"
        },
         "more_than_2_million": {
            "English": "<h3>More than 2 million people are diagnosed with breast cancer each year</h3>",
            "Français": "<h3>Plus de 2 millions de personnes sont diagnostiquées avec un cancer du sein chaque année</h3>"
        },
        "breast_cancer_common": {
            "English": "<h4>Breast cancer is the most common form of cancer globally, and early detection through breast cancer screening can lead to better chances of survival. While screening is critical to improving outcomes, a shortage of specialists around the world means that screening systems are often overburdened, leading to long, anxiety-filled wait times for people awaiting results.</h4>",
            "Français": "<h4>Le cancer du sein est la forme de cancer la plus courante dans le monde, et une détection précoce grâce au dépistage du cancer du sein peut améliorer les chances de survie. Bien que le dépistage soit essentiel pour améliorer les résultats, une pénurie de spécialistes dans le monde entier signifie que les systèmes de dépistage sont souvent surchargés, entraînant de longues attentes anxieuses pour les personnes en attente de résultats.</h4>"
        },
        "ai_system": {
            "English": "<h4>The artificial intelligence-powered system integrates into breast cancer screening workflows to help pathologist identify breast cancer</h4>",
            "Français": "<h4>Le système alimenté par l'intelligence artificielle s'intègre dans les flux de travail de dépistage du cancer du sein pour aider les pathologistes à identifier le cancer du sein</h4>"
        },
        "institute_title": {
            "English": "<h1 style='text-align: left; color:#AA336A;'>Breast Cancer Technology Institute</h1>",
            "Français": "<h1 style='text-align: left; color:#AA336A;'>Institut de Technologie du Cancer du Sein</h1>"
        },
        "ai_detect_breast_cancer": {
            "English": "<h5>AI offers significant benefits in the detection of breast cancer using histopathological images, particularly in the context of biopsies.</h5>",
            "Français": "<h5>L'IA offre des avantages significatifs dans la détection du cancer du sein à l'aide d'images histopathologiques, en particulier dans le contexte des biopsies.</h5>"
        },
        "histopathological_analysis": {
            "English": "<h5>Histopathological analysis involves examining tissue samples under a microscope to identify cancerous cells. AI techniques enhance this process in several ways: </h5>",
            "Français": "<h5>L'analyse histopathologique consiste à examiner des échantillons de tissus au microscope pour identifier les cellules cancéreuses.</h5>"
        },
        "enhanced_image_analysis": {
            "English": "<h3>Enhanced Image Analysis:</h3>",
            "Français": "<h3>Analyse d'image améliorée :</h3>"
        },
        "ai_algorithms_analysis": {
            "English": "<h4>AI algorithms can analyze histopathological images with high precision, detecting subtle features that might be missed by the human eye. </h4>",
            "Français": "<h4>Les algorithmes d'IA peuvent analyser les images histopathologiques avec une grande précision, détectant des caractéristiques subtiles qui pourraient être manquées par l'œil humain.</h4>"
        },
        "consistency_objectivity": {
            "English": "<h3>Consistency and Objectivity:</h3>",
            "Français": "<h3>Consistance et Objectivité:</h3>"
        },
        "ai_consistent_objective_results": {
            "English": "<h4>Unlike human analysis, which can be subjective and vary between pathologists, AI provides consistent and objective results. This uniformity helps in reducing diagnostic errors and ensuring that all patients receive a high standard of care.</h4>",
            "Français": "<h4>Contrairement à l'analyse humaine, qui peut être subjective et varier d'un pathologiste à l'autre, l'IA fournit des résultats cohérents et objectifs. Cette uniformité aide à réduire les erreurs de diagnostic et à garantir que tous les patients reçoivent des soins de haute qualité.</h4>"
        },
        
        "deep_learning_models": {
            "English": "<h4>Our deep learning models, such as the BCX (Xception) model, are designed specifically to detect breast cancer using histopathological images. These advanced models leverage the power of AI to analyze tissue samples with exceptional accuracy, ensuring early and precise detection of breast cancer. By employing the BCX model, we can further improve the diagnostic process, providing pathologists with reliable tools to aid in the fight against breast cancer.</h4>",
            "Français": "<h4>Nos modèles d'apprentissage profond, tels que le modèle BCX, sont spécialement conçus pour détecter le cancer du sein à l'aide d'images histopathologiques. Ces modèles avancés exploitent la puissance de l'IA pour analyser les échantillons de tissus avec une précision exceptionnelle, garantissant une détection précoce et précise du cancer du sein. En utilisant le modèle BCX, nous pouvons améliorer davantage le processus de diagnostic, fournissant aux pathologistes des outils fiables pour les aider dans la lutte contre le cancer du sein.</h4>"
        },
        "model_selection": {
            "English": "Model Selection and Image Upload",
            "Français": "Sélection du modèle et téléchargement de l'image"
        },
        "choose_model": {
            "English": "Choose the model",
            "Français": "Choisissez le modèle"
        },
        "upload_image": {
            "English": "Upload Image",
            "Français": "Télécharger une image"
        },
        "uploaded_image_caption": {
            "English": "Uploaded Image",
            "Français": "Image téléchargée"
        },
        "predict_button": {
            "English": "Predict",
            "Français": "Prédire"
        },
        "prediction_text": {
            "English": "Prediction: {prediction}",
            "Français": "Prédiction : {prediction}"
        },
    
    },
    "texts_statistics": {
        "WHO_intro": {
            "English": "The World Health Organization (WHO) plays a crucial role in monitoring global health trends, offering valuable insights through comprehensive statistical analyses.",
            "Français": "L'Organisation Mondiale de la Santé (OMS) joue un rôle crucial dans la surveillance des tendances mondiales de la santé, offrant des informations précieuses grâce à des analyses statistiques complètes."
        },
        "Chart1_title": {
            "English": "Global Cancer Incidence Rates in 2022",
            "Français": "Taux d'Incidence Globale du Cancer en 2022"
        },
        "Chart1_description": {
            "English": "The bar chart presents the age-standardized incidence rates (per 100,000) of various cancer types for both sexes worldwide in 2022. These statistics, provided by the International Agency for Research on Cancer (IARC) and the World Health Organization (WHO), highlight the prevalence of different cancers across the globe.",
            "Français": "Le graphique en barres présente les taux d'incidence standardisés selon l'âge (pour 100 000) de divers types de cancer pour les deux sexes dans le monde en 2022. Ces statistiques, fournies par l'Agence Internationale de Recherche sur le Cancer (IARC) et l'Organisation Mondiale de la Santé (OMS), soulignent la prévalence des différents cancers à travers le monde."
        },
        "Chart1_breast_cancer": {
            "English": "Breast Cancer: Leading the chart, breast cancer has the highest incidence rate, significantly surpassing other cancer types with an age-standardized rate close to 50 per 100,000. This indicates a widespread occurrence and underscores the need for enhanced screening and prevention efforts.",
            "Français": "Cancer du sein : En tête du classement, le cancer du sein a le taux d'incidence le plus élevé, dépassant de manière significative les autres types de cancer avec un taux standardisé selon l'âge proche de 50 pour 100 000. Cela indique une occurrence généralisée et souligne la nécessité de renforcer les efforts de dépistage et de prévention."
        },
        "Chart2_title": {
            "English": "Global Cancer Mortality Rates in 2022",
            "Français": "Taux de Mortalité Globale du Cancer en 2022"
        },
        "Chart2_description": {
            "English": "The bar chart illustrates the age-standardized mortality rates (per 100,000) of various cancer types for both sexes worldwide in 2022. These statistics, sourced from the International Agency for Research on Cancer (IARC) and the World Health Organization (WHO), provide a clear picture of the deadliest cancers globally.",
            "Français": "Le graphique en barres illustre les taux de mortalité standardisés selon l'âge (pour 100 000) de divers types de cancer pour les deux sexes dans le monde en 2022. Ces statistiques, provenant de l'Agence Internationale de Recherche sur le Cancer (IARC) et de l'Organisation Mondiale de la Santé (OMS), donnent une image claire des cancers les plus meurtriers dans le monde."
        },
        "Chart2_breast_cancer": {
            "English": "Breast Cancer: Breast cancer, with a mortality rate around 15 per 100,000, is the second deadliest cancer. Despite being the most commonly diagnosed cancer, advances in treatment and early detection have helped reduce mortality rates, but it remains a significant cause of cancer deaths among women.",
            "Français": "Cancer du sein : Avec un taux de mortalité d'environ 15 pour 100 000, le cancer du sein est le deuxième cancer le plus mortel. Bien qu'il soit le cancer le plus fréquemment diagnostiqué, les progrès dans le traitement et la détection précoce ont permis de réduire les taux de mortalité, mais il reste une cause significative de décès par cancer chez les femmes."
        }
    }        
}
# Horizontal menu
selected = option_menu(
    menu_title=None,
    options=[texts["menu"]["Home"][language], texts["menu"]["Diagnosis"][language], texts["menu"]["Treatment"][language], texts["menu"]["Statistics"][language]],
    icons=['house', "book", "capsule", "clipboard2-data-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)
if selected == texts["menu"]["Home"][language]:
    # Main title
    logo_path = r'C:\Users\HP 450 G8\Downloads\health_center_logo.png'  # Replace with the actual path to your logo
    if os.path.exists(logo_path):
        col1, col2 = st.columns([1, 8])
        with col1:
            st.image(logo_path, use_column_width=True)
        with col2:
            st.markdown(texts["content"]["institute_title"][language], unsafe_allow_html=True)
    else:
        st.error(f"Error: The file '{logo_path}' does not exist. Please check the file path.")
    
    # Display the health awareness image slider
    health_image_paths = [
        r'C:\Users\HP 450 G8\Downloads\bct.png',  # Replace with the actual paths to your images
        r'C:\Users\HP 450 G8\Downloads\center.jpg'
    ]

    # Image slider
    image_container = st.empty()
    for image_path in health_image_paths:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            new_width = 1500
            new_height = 500
            resized_image = image.resize((new_width, new_height))
            image_container.image(resized_image, use_column_width=False)
            time.sleep(3)  # Display each image for 5 seconds
        else:
            st.error(f"Error: The file '{image_path}' does not exist. Please check the file path.")
    
    st.title(texts["titles"]["about_institute"][language])
    st.markdown(texts["content"]["institute_info"][language], unsafe_allow_html=True)
    path9 = r'C:/Users/HP 450 G8/Downloads/reco.png'
    
    st.image(path9)
    
    col1, col2 = st.columns(2)
    with col1:
        im = Image.open(r'C:\Users\HP 450 G8\Downloads\breast2.png')
        im = im.resize((800, 350))
        st.image(im)

    with col2:
        st.markdown(texts["content"]["more_than_2_million"][language], unsafe_allow_html=True)
        st.markdown(texts["content"]["breast_cancer_common"][language], unsafe_allow_html=True)
    
    col20, col21 = st.columns(2)
    with col20:
        if language == 'English':
            img_path = r'C:\Users\HP 450 G8\Downloads\breast3.png'
        elif language == 'Français':
            img_path = r'C:\Users\HP 450 G8\Downloads\AIBreast.jpg'
        iml = Image.open(img_path)
        iml = iml.resize((800, 250))
        st.image(iml)
    with col21:
        st.markdown(texts["content"]["research_shows"][language], unsafe_allow_html=True)
        st.markdown(texts["content"]["ai_system"][language], unsafe_allow_html=True)
elif selected == texts["menu"]["Diagnosis"][language]:
    #st.title('Artificial intelligence can detect breast cancer missed by doctors')
    st.title(texts["titles"]["diagnosis"][language])
    im3 = Image.open('C:/Users/HP 450 G8/Downloads/breast-detection.png')
    im3 = im3.resize((1500, 500))
    st.image(im3)
    st.markdown(texts["content"]["ai_detect_breast_cancer"][language], unsafe_allow_html=True)
    st.markdown(texts["content"]["histopathological_analysis"][language], unsafe_allow_html=True)
    col22, col23 = st.columns(2)
    with col22:
        iml1 = Image.open(r'C:\Users\HP 450 G8\Downloads\biop.png')
        iml1 = iml1.resize((800, 400))
        st.image(iml1)
    with col23:
        st.markdown(texts["content"]["enhanced_image_analysis"][language], unsafe_allow_html=True)
        st.markdown(texts["content"]["ai_algorithms_analysis"][language], unsafe_allow_html=True)
        st.markdown(texts["content"]["consistency_objectivity"][language], unsafe_allow_html=True)
        st.markdown(texts["content"]["ai_consistent_objective_results"][language], unsafe_allow_html=True)
    st.markdown(texts["content"]["deep_learning_models"][language], unsafe_allow_html=True)
    
    # Image dimensions (adjusted for a more balanced layout)
    img_width, img_height = 224, 224

    # Model paths
    models = {
        "ResNet": r'C:\Users\HP 450 G8\Downloads\my_modelResnet.h5',
        "Xception": r'C:\Users\HP 450 G8\Downloads\my_modelXception.h5'
    }

    # Load and compile the selected model
    def load_model(model_name):
        model_path = models[model_name]
        model = tf.keras.models.load_model(model_path, compile=False)
        model.build((None, img_width, img_height, 3))  # Explicitly set input shape
        return model

    # Preprocess the uploaded image
    def preprocess_uploaded_image(uploaded_image):
        img = Image.open(uploaded_image)
        img = img.resize((img_width, img_height))  # Resize to match the input size of your model
        img = np.array(img) / 255.0  # Normalize the image
        img = img.reshape(-1, img_width, img_height, 3)  # Reshape to match the model's input shape
        return img

    # Make prediction
    def predict_class(model, image):
        prediction = model.predict(image)
        if prediction[0][0] > 0.5:  # Checking the probability of the positive class
            return 'Malignant'
        else:
            return 'Benign'

    # Sidebar for model selection and image upload

    st.header(texts["content"]["model_selection"][language])
    model_name = st.selectbox(texts["content"]["choose_model"][language], list(models.keys()))
    uploaded_image = st.file_uploader(texts["content"]["upload_image"][language], type=["jpg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption=texts["content"]["uploaded_image_caption"][language], use_column_width=True)
        if st.button(texts["content"]["predict_button"][language]):
            model = load_model(model_name)
            image = preprocess_uploaded_image(uploaded_image)
            prediction = predict_class(model, image)
            st.write(texts["content"]["prediction_text"][language].format(prediction=prediction))

if selected == texts["menu"]["Treatment"][language]:
    st.title(texts["titles"]["treatment_stages"][language])

    # Open an image file
    image = Image.open('C:/Users/HP 450 G8/Downloads/treatement.png')
    image = image.resize((1500, 500))
    
    st.image(image)
    # Resize the image
    import random
    quote = random.choice(texts["quotes"][language])
    st.markdown(f"<h4 style='text-align:center;'>{quote}</h4>", unsafe_allow_html=True)
    st.title(texts["titles_treatment"]["BR_surgery"][language])
    col1, col2 = st.columns([2, 3])  # Adjust the ratio as needed
    with col1:
        img = Image.open('C:/Users/HP 450 G8/Downloads/surgery.png')
        img = img.resize((550, 350))
        st.image(img)
    with col2:
        for text in texts["texts_treatment"]["BR_surgery"][language]:
            st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)
    st.title(texts["titles_treatment"]["Rad_therapy"][language])
    # Create two columns for radiation therapy section
    col3, col4 = st.columns([2, 3])
    with col3:
        img2 = Image.open('C:/Users/HP 450 G8/Downloads/radiation.png')
        img2 = img2.resize((550, 400))
        st.image(img2)
        
    with col4:
        for text in texts["texts_treatment"]["Rad_therapy"][language]:
            st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)
    st.title(texts["titles_treatment"]["Che_therapy"][language])
    
    col5, col6 = st.columns([2, 3])  
    with col5:
        img3 = Image.open('C:/Users/HP 450 G8/Downloads/chemotherapy.png')
        img3 = img3.resize((550, 250))
        st.image(img3)
    with col6:
        for text in texts["texts_treatment"]["Che_therapy"][language]:
            st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)
    st.title(texts["titles_treatment"]["Hormone_therapy"][language])
    col7, col8 = st.columns([2, 3])
    with col7:
         img4 = Image.open('C:/Users/HP 450 G8/Downloads/hormon.png')
         img4 = img4.resize((550, 350))
         st.image(img4)
    with col8:
        for text in texts["texts_treatment"]["Hormone_therapy"][language]:
            st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)
    st.title(texts["titles_treatment"]["Targeted_therapy"][language])
    col9, col10 = st.columns([2, 3])
    with col9:
        img5 = Image.open('C:/Users/HP 450 G8/Downloads/t1.png')
        img5 = img5.resize((550, 300))
        st.image(img5)
    with col10:
        for text in texts["texts_treatment"]["Targeted_therapy"][language]:
            st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)
    st.title(texts["titles_treatment"]["Immunotherapy"][language])
    col11, col12 = st.columns([2,3])
    with col11:
        img6 = Image.open('C:/Users/HP 450 G8/Downloads/immunotherapy-.png')
        img6 = img6.resize((550, 200))
        st.image(img6)
    with col12:
        for text in texts["texts_treatment"]["Immunotherapy"][language]:
            st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)
    st.title(texts["titles_treatment"]["Palliative_care"][language])
    col13, col14 = st.columns([2,3])
    with col13:
        img7 = Image.open('C:/Users/HP 450 G8/Downloads/palliative.png')
        img7 = img7.resize((550, 250))
        st.image(img7)
    with col14:
        for text in texts["texts_treatment"]["Palliative_care"][language]:
            st.markdown(f"<h5>{text}</h5>", unsafe_allow_html=True)

if selected == texts["menu"]["Statistics"][language]:
    st.title(texts["titles"]["Statistics_title"][language])
    img10 = Image.open('C:/Users/HP 450 G8/Downloads/st.png')
    img10 = img10.resize((1500, 500))
    st.image(img10)
    
    col40, col41 = st.columns([1, 2])
    with col40:
        img30 = Image.open('C:/Users/HP 450 G8/Downloads/WHO.png')
        img30 = img30.resize((400, 100))
        st.image(img30)
    with col41:
        st.markdown(f"<h3>{texts['texts_statistics']['WHO_intro'][language]}</h3>", unsafe_allow_html=True)
    
    col30, col31 = st.columns(2)
    with col30:
        img20 = Image.open('C:/Users/HP 450 G8/Downloads/chart1.png')
        img20 = img20.resize((1200, 600))
        st.image(img20)
    with col31:
        st.write('')
        st.markdown(f"<h3>{texts['texts_statistics']['Chart1_title'][language]}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>{texts['texts_statistics']['Chart1_description'][language]}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4>{texts['texts_statistics']['Chart1_breast_cancer'][language]}</h4>", unsafe_allow_html=True)
    
    col32, col33 = st.columns(2)
    with col32:
        img60 = Image.open('C:/Users/HP 450 G8/Downloads/chart2.png')
        img60 = img60.resize((1200, 600))
        st.image(img60)
    with col33:
        st.markdown(f"<h3>{texts['texts_statistics']['Chart2_title'][language]}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>{texts['texts_statistics']['Chart2_description'][language]}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4>{texts['texts_statistics']['Chart2_breast_cancer'][language]}</h4>", unsafe_allow_html=True)

        