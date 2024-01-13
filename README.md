## Projet IA 

# Objectif
- Le but du projet était la détection de différents éléments sur une vidéo par l'intermédiaire de YOLO et de les flouter.
- Le projet utilise donc yoloV8 par Ultralytics et Robotflow pour les datasets.
- Plusieurs difficultés ont été rencontrées sur ce projet.
- La principale venait de la détection en elle même, dû à un manque de moyens matériels, le "training" de l'IA s'est avéré long et compliqué.
- Résultant de ces difficultés, l'IA ne peut effectuer que de la détection d'objets simple et non de la segmentation.
- L'IA peut détecter les visages, les plaques d'immatriculation et les vitres de voiture.
- Malheureusement, cette identification est souvent approximative.

# Projet
- Ce projet a plusieurs fichiers.
- Des vidéos, des images, 2 modèles et du code python.
- yolo8n-seg.pt : Un modèle officiel permettant la segmentation.
- yoloUni.pt : Notre modèle permettant la reconnaissance des visages, plaques et fenêtres (peu performant, le data set que nous avons fait était trop petit).
- Lors du "train", différentes données sur les performances du modèle sont créés, elles sont stockées de le répertoire "training_data"
- yoloTrain.py : Contient toutes les informations nécessaires pour créer un modèle, l'entraîner sur un dataset Roboflow et prédire sur une image.
- yoloSeg2.py : Prototype fonctionnant sur un modèle de segmentation et permet un floutage plus propre, n'a pas pu être utilisé pour le floutage de visage uniquement ou de plaque ou de vitre. Cette version floute les "personnes" et les "bus" en utilisant "yolo8n-seg.pt". L'identification est plus poussée mais le modèle n'est pas de nous.
- yoloBox2.py : Prototype fonctionnant sur notre modèle de détection et permet un floutage approximatif peut être utilisé pour les visages, plaque d'immatriculation, fenêtre fermé de véhicule.
- App.py : Une IHM où l'utilisateur peut selectionner une vidéo qui sera floutée en direct, des boutons sont disponibles pour filtrer le contenu à flouter (ils sont activés par défaut) ou ou changer de modèle (notre modèle par defaut).
- Cette commande permet d'utiliser la prediction directement depuis un terminal : "yolo detect predict model=yoloUni.pt source='./images/human.jpg'"
- Si on test la commande suivante : "yolo segment predict model=yolov8n-seg.pt source='./images/plaque.jpg'" on peut voir que le model de ultralytics ne permet pas la segmentation de plaque d'immatriculation.