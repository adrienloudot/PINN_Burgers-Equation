# Résolution de l'équation de Burgers par réseau de neurones informé par la physique (PINN)

Implémentation PyTorch d'un Physics-Informed Neural Network (PINN) pour approcher la solution de l'équation de Burgers visqueuse, avec validation quantitative par solution exacte et stratégie d'entraînement par curriculum temporel et pondération causale.

---

## L'équation de Burgers

$$u_t + u\, u_x = \nu\, u_{xx}, \quad x \in [-1,1],\; t \in [0,1]$$

avec :
- **Condition initiale :** $u(x, 0) = -\sin(\pi x)$
- **Conditions aux bords :** $u(-1, t) = u(1, t) = 0$
- **Viscosité :** $\nu = 0{,}01/\pi \approx 3{,}18 \times 10^{-3}$

Cette équation est le prototype des EDP non-linéaires en mécanique des fluides : elle combine un terme d'advection non-linéaire $u\,u_x$ (qui tend à raidir le profil jusqu'à former un choc) et un terme diffusif $\nu\, u_{xx}$ (qui lisse la solution). La valeur $\nu = 0{,}01/\pi$ est délibérément petite : la solution développe un gradient quasi-discontinu en $x=0$ vers $t \approx 1$, ce qui en fait un benchmark difficile et standard pour les méthodes numériques.

---

## Principe des PINNs

L'idée est de représenter $u(x,t)$ par un réseau de neurones $u_\theta(x,t)$ et d'encoder la physique directement dans la fonction de coût, sans discrétiser le domaine en grille. La loss combine trois termes :

$$\mathcal{L}(\theta) = \lambda_\text{pde}\,\mathcal{L}_\text{pde} + \lambda_\text{ic}\,\mathcal{L}_\text{ic} + \lambda_\text{bc}\,\mathcal{L}_\text{bc}$$

- $\mathcal{L}_\text{pde}$ : résidu moyen au carré de l'EDP, calculé par **différentiation automatique** (autograd PyTorch) sur des points de collocation tirés aléatoirement dans le domaine
- $\mathcal{L}_\text{ic}$ : erreur quadratique sur la condition initiale
- $\mathcal{L}_\text{bc}$ : erreur quadratique sur les conditions aux bords

Les dérivées $\partial_t u_\theta$, $\partial_x u_\theta$, $\partial_{xx} u_\theta$ sont **exactes** (pas de différences finies), ce qui est la force principale de l'approche.

---

## Démarche et difficultés rencontrées

### 1. Implémentation de base

Le réseau est un MLP avec activation tanh et initialisation Xavier, entraîné en deux phases : **Adam** (exploration) suivi de **L-BFGS** (affinage, compatible avec les dérivées d'ordre 2). La structure modulaire du projet sépare clairement configuration, modèle, physique, entraînement et évaluation.

### 2. Validation par solution exacte — transformée de Cole-Hopf

Pour valider quantitativement le PINN, on calcule la solution exacte de l'équation de Burgers via la transformée de Cole-Hopf, qui ramène l'EDP non-linéaire à une équation de la chaleur. La solution s'écrit :

$$u(x,t) = \frac{2\nu \,\partial_x \phi}{\phi}, \qquad \phi(x,t) = \int_{-\infty}^{+\infty} \exp\!\left[\frac{\cos(\pi\xi)}{2\nu\pi} - \frac{(x-\xi)^2}{4\nu t}\right] d\xi$$

Le calcul numérique de cette intégrale a nécessité une **stabilisation en log-space** (trick log-sum-exp) : pour $\nu$ petit, l'exponentielle atteint des valeurs de l'ordre de $e^{50}$, provoquant un overflow en virgule flottante standard. On soustrait le maximum de l'exposant avant d'exponentier ; ce décalage s'annule dans le rapport $\phi_x / \phi$.

### 3. Échantillonnage adaptatif

Le choc de Burgers se forme dans la région $x \approx 0$, $t \in [0{,}5, 1]$. Un tirage uniforme des points de collocation sous-échantillonne cette zone critique. On utilise donc un mélange : 75 % de points uniformes sur tout le domaine, 25 % de points concentrés autour du choc.

### 4. Curriculum temporel

Un entraînement direct sur $t \in [0,1]$ conduit à un minimum local : le réseau apprend une solution stationnaire qui satisfait les bords mais pas la dynamique. On adopte une stratégie de **curriculum temporel** inspirée de la littérature récente : la fenêtre temporelle d'entraînement croît progressivement en quatre stages, forçant le réseau à apprendre la dynamique lisse avant de s'attaquer au choc.

| Stage | Fenêtre | Epochs Adam | Remarque |
|-------|---------|-------------|----------|
| 1 | $t \in [0, 0{,}25]$ | 3 000 | Sinus lisse, sans choc |
| 2 | $t \in [0, 0{,}50]$ | 5 000 | Choc qui commence à se former |
| 3 | $t \in [0, 0{,}75]$ | 5 000 | Choc bien établi |
| 4 | $t \in [0, 1{,}00]$ | 10 000 | Domaine complet |

### 5. Pondération causale (Wang et al., 2022)

Même avec le curriculum, le résidu PDE reste élevé dans les premiers instants car l'optimiseur traite tous les points de collocation de façon équivalente, sans respecter l'ordre causal de la dynamique (c'est-à-dire que le réseau cherche à minimiser de manière homogène sur tout le domaine, sans nécessairement privilégier les zones de choc). Pour tenter de résoudre ce problème, on implémente la **causal weighting** (Wang et al., 2022) : chaque point de collocation au temps $t_k$ reçoit un poids

$$w(t_k) = \exp\!\left(-\varepsilon \sum_{t_j < t_k} \bar{\mathcal{L}}_j\right)$$

où $\bar{\mathcal{L}}_j$ est le résidu moyen dans la tranche temporelle $j$. Ainsi, les temps tardifs ne contribuent à la loss que lorsque les temps antérieurs sont bien résolus. Dans les logs d'entraînement, le poids moyen $w$ monte de $\approx 0{,}03$ en début de stage 4 jusqu'à $\approx 0{,}99$ après L-BFGS, confirmant que la causalité est progressivement respectée.

---

## Architecture et hyperparamètres

```
Entrée (x, t)  →  Linear(2 → 128)  →  tanh
               →  Linear(128 → 128) →  tanh   × 3
               →  Linear(128 → 1)              (sortie linéaire)
```

| Paramètre | Valeur |
|-----------|--------|
| Paramètres entraînables | 50 049 |
| Points de collocation $N_f$ | 20 000 |
| Points CI $N_i$ | 1 000 |
| $\lambda_\text{ic}$ | 20 |
| $\lambda_\text{bc}$ | 10 |
| $\varepsilon$ (causal) | 1{,}0 |
| lr Adam | $10^{-3}$ |
| L-BFGS max iter (stage 4) | 500 |

---

## Structure du projet

```
pinn_burgers/
├── config.py        # hyperparamètres centralisés
├── model.py         # réseau PINN (MLP + initialisation Xavier)
├── physics.py       # résidu PDE, pondération causale, génération des données
├── train.py         # boucle Adam + L-BFGS avec curriculum
├── evaluate.py      # solution exacte Cole-Hopf, métriques, figures
├── main.py          # point d'entrée (skip_training pour réévaluer sans réentraîner)
└── outputs/         # poids, métriques, figures (ignoré par git)
```

---

## Lancement

```bash
pip install -r requirements.txt
python main.py
```

Les sorties sont écrites dans `outputs/` :
- `model.pt` — poids entraînés
- `training_history.csv` — loss par epoch
- `metrics.csv` — erreurs L2 et L∞ relatives finales
- `heatmaps.png` — comparaison prédiction / exacte / erreur absolue
- `slices.png` — coupes temporelles à $t = 0{,}25$, $0{,}50$, $0{,}75$
- `loss_history.png` — courbes d'entraînement par stage et par phase

Pour réévaluer un modèle déjà entraîné sans relancer l'optimisation :
```python
# config.py
skip_training: bool = True
```

---

## Résultats et comparaison avec la littérature

| Métrique | Valeur obtenue |
|----------|----------------|
| Erreur L2 relative | $\approx 0{,}78$ |
| Erreur L∞ relative | $\approx 0{,}96$ |

La solution PINN capture correctement la condition initiale (erreur $< 0{,}01$) et les conditions aux bords (erreur $< 10^{-6}$). Elle reproduit qualitativement la dynamique de Burgers (formation du choc, antisymétrie) mais présente un **décalage de phase spatial** qui s'accumule progressivement entre $t=0$ et $t=1$.

### Comparaison avec Raissi et al. (2019)

Raissi et al. proposent deux formulations pour Burgers. Dans leur **modèle continu** (le plus proche de celui-ci), ils rapportent des erreurs L2 de l'ordre de $10^{-2}$ à $10^{-3}$, mais en travaillant avec $\nu = 0{,}1/\pi$ — une viscosité **dix fois plus grande**, qui atténue considérablement le choc. Leur **modèle discret** (Runge-Kutta implicite à 100 stages) appliqué à $\nu = 0{,}01/\pi$ obtient de meilleures performances en résolvant le problème séquentiellement en temps, ce qui contourne naturellement le problème de causalité.

La comparaison directe est alors à prendre avec dur recul : le cas $\nu = 0{,}01/\pi$ en formulation continue est significativement plus difficile, et les erreurs observées ici sont cohérentes avec ce que la littérature post-2019 documente pour ce régime.

### Pourquoi le décalage persiste

Un MLP à activation tanh est une fonction **infiniment dérivable** ; or la solution de Burgers avec petit $\nu$ devient quasi-discontinue en $x = 0$ vers $t = 1$. Le réseau ne peut pas représenter exactement cette discontinuité, il l'approche par une fonction lisse dont le centre est décalé. Cela ne semble donc pas être un problème d'optimisation (car la loss converge bien) mais une **limite de capacité de représentation** de l'architecture utilisée.

---

## Pistes d'amélioration

**Architectures adaptées aux discontinuités**
- Activations **SIREN** ($\sin$) ou **adaptatives** (apprenables par couche) permettent de représenter des fréquences spatiales plus élevées
- Les **réseaux hp-PINN** raffinent localement le réseau près du choc, comme un maillage adaptatif

**Formulation temporelle discrète**
- Reproduire l'approche Runge-Kutta de Raissi et al. : résoudre séquentiellement sur des sous-intervalles temporels avec un schéma implicite d'ordre élevé, ce qui garantit la causalité par construction

**Intégration de données d'observation**
- Ajouter un terme $\mathcal{L}_\text{obs}$ pénalisant l'écart à des mesures synthétiques issues de la solution exacte (problème inverse), ce qui ancre le réseau dans la bonne solution et réduit l'espace des minima locaux

**Pondération causale adaptative**
- Faire varier $\varepsilon$ au cours de l'entraînement (grand au début, petit à la fin) plutôt que de le fixer, pour combiner la rigueur causale initiale avec une convergence globale en fin d'optimisation

---

## Références

- Raissi, M., Perdikaris, P., Lagaris, I.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Wang, S., Sankaran, S., Perdikaris, P. (2022). *Respecting causality is all you need for training physics-informed neural networks.* arXiv:2203.07404.
