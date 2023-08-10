import os
import random

# Chemin vers le dossier contenant les images et les labels
dossier_images = 'C:\\Users\\AdminODC\\Documents\\BIONIX\\obj'

# Ratio pour diviser les données en train et test
ratio_train = 0.8  # 80% des données pour l'entraînement, 20% pour les tests

# Liste pour stocker les chemins d'accès aux images
chemins_images = []

# Parcourir le dossier des images
for root, _, files in os.walk(dossier_images):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            chemin_image = os.path.join(root, file)
            chemins_images.append(chemin_image)

# Mélanger aléatoirement la liste des chemins d'accès aux images
random.shuffle(chemins_images)

# Diviser les données en train et test
indice_separation = int(ratio_train * len(chemins_images))
chemins_train = chemins_images[:indice_separation]
chemins_test = chemins_images[indice_separation:]

# Écrire les chemins d'accès dans les fichiers train.txt et test.txt
with open('train.txt', 'w') as f_train:
    for chemin in chemins_train:
        f_train.write(chemin + '\n')

with open('test.txt', 'w') as f_test:
    for chemin in chemins_test:
        f_test.write(chemin + '\n')

print("Fichiers train.txt et test.txt créés avec succès.")
