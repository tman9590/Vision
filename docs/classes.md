# Animal Classes Reference

| ID | Class | Colour | Notes |
|---|---|---|---|
| 0 | bear | `#8B0000` | Black, brown, polar |
| 1 | bird | `#1E90FF` | Generic — works on most species |
| 2 | cat | `#FF69B4` | Domestic + bobcat |
| 3 | cow | `#8B4513` | Cattle, may generalise to bison |
| 4 | deer | `#228B22` | Whitetail, mule deer |
| 5 | dog | `#FFA500` | All breeds |
| 6 | elephant | `#708090` | African & Asian |
| 7 | fox | `#FF4500` | Red fox — limited training data |
| 8 | horse | `#D2691E` | Horse, pony |
| 9 | monkey | `#9ACD32` | Various primates |
| 10 | mouse | `#A9A9A9` | Mouse, rat |
| 11 | raccoon | `#4B0082` | Common raccoon |
| 12 | rabbit | `#FFD700` | Domestic & wild |
| 13 | sheep | `#CCCCCC` | Sheep, goat |
| 14 | skunk | `#2F4F4F` | Striped skunk — limited data |
| 15 | squirrel | `#CD853F` | Grey & red squirrel |
| 16 | wolf | `#6A5ACD` | Grey wolf |
| 17 | coyote | `#DAA520` | May be confused with wolf/dog at distance |
| 18 | turkey | `#B8860B` | Wild turkey |
| 19 | duck | `#008B8B` | Mallard & common ducks |

## Known Limitations

- **fox / skunk / coyote / turkey** — fewer training examples; lower recall expected
- **bird** — coarse class; won't distinguish species
- **dog vs. coyote vs. wolf** — visually similar at low resolution; use zone-based automations

## Improving Specific Classes

Add 500–1000 annotated images for the target class to `data/train/` then re-run `scripts/train.py`.
