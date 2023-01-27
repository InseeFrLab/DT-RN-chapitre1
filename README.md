# Chapitre 1 : Apprentissage algorithmique (*Machine learning*) et réseaux de neurones, concepts et prise en main

[![Onyxia](https://img.shields.io/badge/Launch-Datalab-orange?logo=R)](https://datalab.sspcloud.fr/launcher/ide/rstudio?autoLaunch=false&onyxia.friendlyName=%C2%ABdt-rn-chapitre1%C2%BB&security.allowlist.enabled=false&service.image.custom.enabled=true&service.image.pullPolicy=%C2%ABAlways%C2%BB&service.image.custom.version=%C2%ABinseefrlab%2Fdt-rn-chapitre1%3Alatest%C2%BB)
[![Build](https://img.shields.io/github/actions/workflow/status/ThomasFaria/DT-RN-chapitre1/build-image.yaml?label=Build
)](https://hub.docker.com/repository/docker/inseefrlab/dt-rn-chapitre1)

Le répertoire contient l'ensemble des programmes du chapitre 1.

## Prise en main
L'ensemble du codes sources utilisés dans ce chapitre est accompagné de son [image docker](https://hub.docker.com/repository/docker/inseefrlab/dt-rn-chapitre1) pour assurer une totale reproductibilité des résultats.

Celle-ci peut être utilisée pour vous éviter de télécharger les dépendances nécessaires à ce chapitre. Vous pouvez la récupérer avec la commande suivante :

```
docker pull inseefrlab/dt-rn-chapitre1:latest
```

Nous vous recommendons fortement l'utilisation d'[Onyxia](https://github.com/InseeFrLab/onyxia-web), la plateforme *datascience* développé par l'[Insee](https://www.insee.fr/fr/accueil)). Pour ce faire vous pouvez suivre ces étapes :

- Etape 0: Allez sur [https://datalab.sspcloud.fr/home](https://datalab.sspcloud.fr/home). Cliquer sur **Sign In** et ensuite **create an account** avec votre adresse email institutionnelle ou académique.
- Etape 1: Cliquez [ICI](https://datalab.sspcloud.fr/launcher/ide/rstudio?autoLaunch=true&onyxia.friendlyName=«dt-chap4»&security.allowlist.enabled=false&service.image.custom.enabled=true&service.image.pullPolicy=«Always»&service.image.custom.version=«inseefrlab%2Fdt-rn-chapitre4») ou sur le badge orange en haut de la page pour lancer un service.
- Etape 2: **Ouvrez** le service et suivez les instructions affichées concernant l'**identifiant** et le **mot de passe**.
- Etape 3: **Créez un nouveau projet** et **clonez** le code grâce à la commande suivant : ```git clone https://github.com/ThomasFaria/DT-RN-chapitre1.git```.

Tous les packages ont déjà été installés dans l'image docker et les dépendances sont gérées grâce au package `renv`. Il suffit d'executer la commande `renv::restore()` afin de réinstaller tous les packages dans votre service, comme déclaré dans le lockfile.

## Organisation

*Les programmes sont en **R***

Le script principal [neural_network_cifar100.R](https://github.com/ThomasFaria/DT-RN-chapitre1/blob/main/R/neural_network_cifar100.R) contient le déroulé intégral de l'exemple de classification de meubles (chaises et tables) sur la base d'images *CIFAR100* c'est-à-dire :
1. L'import des images à utiliser
2. Un premier exemple de réseau de neurones avec la librairie **neuralnet**
3. Une classification par une régression pénalisée (Lasso)
4. Un réseau de neurones plus élaboré avec la librairie **Keras**

Le script [neural_network_cifar100_pytorch.R](https://github.com/ThomasFaria/DT-RN-chapitre1/blob/main/R/neural_network_cifar100_pytorch.R) le réseau de neurones du 4. précédent programmé avec **torch**

Le script [Un_premier_convnet.R](https://github.com/ThomasFaria/DT-RN-chapitre1/blob/main/R/Un_premier_convnet.R) effectue la classification des chaises et des tables de la base **CIFAR100** à l'aide d'un **convnet**. Cet exemple est à relier au chapitre 3 sur les parcelles cadastrales.

Les codes ont été écrits par [Stéphanie Himpens](https://github.com/srhimp) (Insee / Banque de France).
