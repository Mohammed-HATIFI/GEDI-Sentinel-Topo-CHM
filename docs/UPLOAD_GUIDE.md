# Publication GitHub et Zenodo

## Titres officiels

- GitHub: *Multisource canopy-height mapping across three contrasting Moroccan forest landscapes: aggregate error versus height-dependent performance under GEDI supervision*
- Zenodo: *Derived data, trained models, and canopy-height maps for GEDI-Sentinel canopy-height mapping in Morocco*

## GitHub

Depuis PowerShell:

```powershell
Set-Location 'C:\Users\Dell\Desktop\GEDI-Sentinel-CHM'
python scripts\check_release.py
git status --short
git add README.md CITATION.cff scripts\check_release.py notebooks docs results\metrics
git diff --cached --stat
git commit -m "Prepare v1.0.0 reproducibility release"
git push origin HEAD
git tag -a v1.0.0 -m "GEDI-Sentinel-CHM v1.0.0"
git push origin v1.0.0
```

Inspecter `git status` et `git diff --cached` avant le commit. Ne pas ajouter les donnees brutes, les sorties intermediaires, les secrets ou les checkpoints non destines a la diffusion.

## Zenodo

1. Creer un nouvel upload sur Zenodo et choisir le type `Dataset`.
2. Televerser l'archive ZIP finale et verifier son SHA-256.
3. Utiliser le titre Zenodo officiel ci-dessus, renseigner les auteurs, l'affiliation, la description, les mots-cles et la licence des donnees.
4. Ajouter l'URL GitHub comme identifiant relie et indiquer que le depot contient le code source associe.
5. Reserver le DOI avant publication, puis reporter ce DOI dans le manuscrit, `README.md` et `CITATION.cff`.
6. Verifier l'aperçu et publier seulement apres le controle final du contenu.

La licence du code est MIT. Les donnees derivees doivent conserver leur licence annoncee dans le depot Zenodo; les entrees tierces non redistribuables ne doivent pas etre incluses.
