# Chanka Quechua orthography spec for MT normalization

**Purpose.** Single citable reference governing the normalization of Spanish↔Chanka Quechua training data. Every rule below is sourced; this is the source of truth that `scripts/chanka_normalizer.py` implements.

**Scope.** Chanka (Ayacucho-Chanka) Quechua, ISO 639-3 `quy`. The Cusco-Collao variety (`quz`) is referenced only to enumerate what Chanka is NOT.

**Methodological position.** The most recent published low-resource Spanish-Quechua MT system (Dhawan et al., LoResMT 2026, §3.3) applies "deterministic orthographic normalization" that addresses *only* four spacing artifacts (`ch a...→cha...`, `sin ch i→sinchi`, etc.) and explicitly states this is done *"without relying on external linguistic resources."* This is a methodological gap. The spec below closes it by grounding every rule in the canonical MINEDU 2021 manual, verified against Soto Ruiz 1976 and the trivocalist consensus initiated by RM N° 1218-85-ED.

---

## 1. Authority chain

| Source | Date | Role |
|---|---|---|
| **RM N° 1218-85-ED** | 18 Nov 1985 | Original Peruvian government decree establishing the trivocalist (`a, i, u`) official Quechua alphabet. Suppressed `<e>` and `<o>` which the 1975 RM 4023-75-ED had included. |
| **Cerrón-Palomino, *Lingüística Quechua*** (Centro Bartolomé de las Casas) | 1987 | Trivocalism academic foundation. ⚠ Note: his 1994 "Unified Southern Quechua" dictionary *preserves* Cusco laryngeals even for Ayacucho readers — MINEDU and Soto disagree on this point. |
| **Soto Ruiz, *Gramática Quechua: Ayacucho-Chanca*** (IEP/MINEDU) | 1976 | The native-speaker-authored grammar that codifies the 3-vowel, 15-consonant Ayacucho-Chanka inventory. **No** aspirated, **no** ejective series. Matches modern MINEDU. |
| **MINEDU, *Urin Qichwa Qillqay Yachana Mayt'u / Manual de escritura quechua sureño*** (Chuquimamani Valer, Chávez Gonzales, Riveros Paravicino, Jara Luna, Cárdenas Guzmán, Quintasi Mamani) | 2021 | **Canonical reference.** Government-mandated, EIB-distributed orthographic manual. Defines the Chanka 18-grapheme alphabet, the laryngeal-Collao-only rule, and provides ~80 explicit *wrong → right* examples. |
| **PROEIB-Andes** (post-1985 publications) | ongoing | Aligned with the MINEDU/Cerrón school. |
| **AMLQ** (Academia Mayor de la Lengua Quechua, Cuzco) | ongoing | ⚠ **Pentavocalist outlier.** Cuzco-centric; not authoritative for Chanka. Listed only to flag that any text citing AMLQ likely violates MINEDU. |

**Operating principle.** When sources diverge (specifically, Cerrón-Palomino 1994 vs Soto/MINEDU on laryngeal retention), **we follow MINEDU 2021 because (a) it is the standard the AmericasNLP 2021 quy test set follows, (b) it is what Chuya Qellqa Bible follows, and (c) it is the explicit current pedagogical norm.**

---

## 2. The Chanka alphabet (locked)

The Southern Quechua alphabet has 28 graphemes total (3 vowels + 25 consonants). Of these, only **18 are Chanka**:

### Chanka graphemes (18)

**Vowels (3):** `a`, `i`, `u`

**Consonants (15):** `ch`, `h`, `k`, `l`, `ll`, `m`, `n`, `ñ`, `p`, `q`, `r`, `s`, `t`, `w`, `y`

### Graphemes FORBIDDEN in Chanka

| Class | Graphemes | Reason |
|---|---|---|
| Aspirated digraphs | `chh`, `kh`, `ph`, `qh`, `th` | Collao-only laryngealized phonemes |
| Glottalized + apostrophe | `ch'`, `k'`, `p'`, `q'`, `t'` | Collao-only ejectives |
| Spanish-pentavocalist | `e`, `o` | Allophonic in Chanka; written `i`, `u` |
| Spanish-introduced | `b`, `d`, `f`, `g`, `j`, `v`, `x`, `z` | Not in MINEDU alphabet (§2.5.3) |
| Spanish digraphs as `/k/` | `c` (not in `ch`), `cc`, `qu` | Mission-era spellings; replace with `k` |
| Standalone apostrophe `'` | (any use other than Collao ejective) | Never used in Chanka writing |

**Citation:** *Manual de escritura quechua sureño* p. 44 (alphabet), p. 47–48 (vowels), p. 61 §2.5.3 (forbidden letters).

---

## 3. The seven core rules

### Rule R1 — Strip apostrophes

Every standalone apostrophe inside or adjacent to a Quechua token must be removed (the only orthographic function of `'` in Southern Quechua is to mark Collao ejective stops; Chanka has no ejectives).

- `p'acha → pacha`, `t'anta → tanta`, `mayt'u → maytu`, `q'illu → qillu`, `k'aspi → kaspi`, `ch'uklla → chuklla`, `sach'a → sacha`, `wat'a → wata`, `hap'iy → hapiy`, `q'apiy → qapiy`, `mut'i → muti`, `wik'uña → wikuña`, `llamk'ay → llamkay`, `mink'a → minka`, `siq'uy → siquy`, `wirp'a → wirpa`, `samp'a → sampa`, `p'inqay → pinqay`, `p'uchqu → puchqu`, `hayk'ap → haykap`, `huk'ucha → ukucha`, `wichq'ay → wichqay`, `hump'i → humpi`, `k'umu → kumu`, `k'ispiñu → kispiñu`, `hak'akllu → akakllu`, `mach'ay → machay`, `q'iwiy → qiwiy`, `hich'ay → hichay`, `wayq'u → wayqu`, `surq'an → surqan`

### Rule R2 — De-aspirate digraphs

Aspirated `Ch`, `K`, `P`, `Q`, `T` written `chh, kh, ph, qh, th` must collapse to their plain forms.

- `chh → ch`, `kh → k`, `ph → p`, `qh → q`, `th → t`
- `phullu → pullu`, `thuqay → tuqay`, `chhalla → chala`, `qhari → qari`, `mikhuy → mikuy`, `khuchi → kuchi`, `llanthu → llantu`, `qhaway → qaway`, `phukuy → pukuy`, `raphi → rapi`, `saphi → sapi`, `aqha → aqa`, `ichhu → ichu`, `phatma → patma`, `thuñiy → tuñiy`, `uphakuy → upakuy`, `qhachun → qachun`, `khuyapayay → kuyapayay`, `ñaqha → ñaqa`, `rikch'aq` after R1 + R2 → `rikchaq`

### Rule R3 — Enforce 3-vowel system

In Quechua tokens, `e → i` and `o → u`. The MINEDU manual is explicit that `[e]` and `[o]` are conditioned allophones (next to uvulars `q`, `qh`, `q'`); they must never be written.

- `ñoqa → ñuqa`, `qelqay → qillqay`, `qosqo → qusqu`, `qhepa → qipa` (after R2: `qipa`), `sonqo → sunqu`, `qollqe → qullqi`, `q'ello → qillu` (after R1: `qillu`), `onqoy → unquy`, `tukoy → tukuy`, `lloqsiy → lluqsiy`, `noqa → ñuqa`, `qopa → qupa`, `qom → qum`, `oqa → uqa`, `urqo → urqu`, `qechwa → qichwa`, `hampeq → hampiq`

**Carve-out:** Do NOT apply Rule R3 to Spanish loanwords, proper nouns, or Spanish text spans. See §5.

### Rule R4 — Replace forbidden graphemes

- `j → q` or `h` or `k` (resolved by per-token wordlist in §7.4 — DO NOT guess from context, use the table):
  - `jam → qam`, `jucha → hucha`, `jampi → hampi`, `jawapi → hawapi`, `jatun → hatun`, `huj → huk`, `jallpa → allpa`, `mihuna/mijuna → mikuna`
- `c` (not in `ch`) and `cc` and `qu` (as `/k/`) → `k` or `q`:
  - `camay → kamay`, `quilla → killa`, `ccapacc → qapaq`, `ccollque → qullqi`
- `f → p` (Chanka has no `/f/`):
  - `fukuy → pukuy`, `chhafchiy → chapchiy`
- `b, d, g, v, x, z, sh` flagged as non-canonical (preserve in Spanish text; replace by spec mapping in Quechua tokens):
  - `sh → s` or `ch` per canonical wordlist (no general rule; manual is explicit there is none)
  - `pishqa, pisqa → pichqa`; `ashka, askha → achka`; `washka, wachka → waska`; `kuchka, kushka → kuska`

### Rule R5 — Diphthong breaking with `w`/`y`

Spanish-influenced vowel clusters must be broken:

- `hua → wa`, `hue → wi`, `huo → wu` (then R3), `qui → ki`, `cu → ku`, `gua → wa`, `ai → ay`, `ei → iy`, `au → aw`
- `huaita → wayta`, `maitu → maytu`, `huauqei → wawqiy`, `yahuar → yawar`, `huata → wata`, `huaina → wayna`, `ccatimuhuay → qatimuway`
- VV inside a syllable is forbidden: `*wasy → wasi`, `*misky → miski`

### Rule R6 — `<ll>` before uvular `<q>`

When `/λ/` precedes `/q/` at syllable boundary, write `<ll>` not `<l>`:

- `qulqi → qullqi`, `walqa → wallqa`, `salqa → sallqa`, `alqu → allqu`, `lliqlla → lliklla`

### Rule R7 — `<m>` vs `<n>` before `<p>`

- Root-internal `/m/` before `/p/`: keep `<m>`. `pampa`, `tampa`, `tampay`, `kimsa`.
- 3P possessor suffix `-n` before a `p`-initial suffix: keep `<n>` even if pronounced `[m]`. `wasinpi`, `llaqtanpi`, `wampunpaq`.
- Common errors: `qan → qam` (2sg pronoun ends in `m`); `allim → allin`; `wasimpi → wasinpi`; `t'anpa → tampa`.

---

## 4. Structural rules

### S1 — Suffix attachment

Quechua is agglutinative; all suffixes attach to the root without space and without hyphen. The hyphen `-` is reserved for morphological annotation in grammars and bulleted lists; it never appears in running text.

- `wasi-pi` written `wasipi`; `mama-y-qa` written `mamayqa`; `tusu-nqa-ku-raq-mi` written `tusunqakuraqmi`; `uyarinakusunchik`.

### S2 — Preserve full suffix forms even when contracted in speech

- `[wasiki]` → write `wasiyki` (the `-y-` is preserved).
- `[nchis] / [nchex] / [nchi]` → always written `-nchik`.
- `mamanchis → mamanchik`.

### S3 — Genitive in Chanka is `-pa`

After both vowels and consonants. The Collao genitive after vowels is `-p`; the slash-unified form is `warmip/warmipa` where the Chanka reading is the right-hand side.

### S4 — Use `-ntin`, not `*-puwan` / `*-piwan`, for "X with me too"

- ✓ `Ñuqantin risunchik`
- ✗ `Ñuqapuwan risunchik` / `Ñuqapiwan risunchik`

### S5 — Compound words are written separated

The two roots are kept as separate words, each capitalized if a proper noun:

- `yachay wasi` (school), `chaki taklla` (foot-plough), `Pacha Mama`, `Tayta Inti`, `Aya Kuchu` (Ayacucho), `Wanka Willka` (Huancavelica), `Apu Rimaq` (Apurímac), `Machu Pikchu`, `Pacha Chaka`.

### S6 — Tilde / acute accent only on listed cases

Default: no tilde on running words (Chanka is penultimate-stress, no need to mark).

Use acute accent ONLY for:

- `arí` (yes — vs `ari` "pues")
- vocatives: `¡Tayllalláy!`, `¡Urpilláy!`
- interjections: `¡Alaláw!`, `¡Achakáw!`, `¡Achaláw!`, `¡Atatáw!`, `¡Añañáw!`
- suffixes `-chá / -yá / -má` (fused from "Rz. + ari"): `purinqachá`, `hamuyá`, `manamá`.

### S7 — Capitalization

- Sentence-initial; proper nouns of people, animals, named natural elements; divinities (`Pacha Mama`, `Apu`, `Tayta Inti`, `Wiraqucha`); place names (`Aya Kuchu`, `Qusqu`, `Titi Qaqa`); Andean ceremonial months (`Inti Raymi`, `Pawqar Waray`, `Aymuray`).
- Two-root proper nouns: both capitalized, separated (`Awqa Tinku`).
- Digraphs/trigraphs: only the first letter capitalized (`Ch`, `Ll`).

### S8 — Punctuation

- Paired `¿…?` and `¡…!`. The interrogative suffix `-chu` does NOT replace `¿…?`.
- Em-dash `—` for dialogue (glued to first speaker word). No period after `?`/`!`.
- Possession by suffix `-pa`/`-p`, never apostrophe `'s`.
- The slash `/` and tilde `~` are auxiliary signs (variant-form unifier, synonym marker) — never appear in running text outside dictionary-like contexts.

---

## 5. Spanish loanwords and proper nouns

### L0 — Operative policy: ONE canonical form per concept (DECISIVE)

**Problem.** Spanish loanwords are written inconsistently across quy training data: `computadora` / `kumputarura` / `computadora nisqa` all attested. MINEDU itself is internally inconsistent (uses both `kumputarura` and `kumputatura` in the same vocab). A model trained on a mix produces inconsistent outputs (v32 demonstrated this).

**Decision.** For every Spanish concept, the canonical form is **fixed and uniquely determined** by this decision tree:

```
For each Spanish concept C:
  1. If a NATIVE Chanka word for C exists (e.g., "kuraq" for "mayor") → use the NATIVE word.
  2. Else if C is in the L1b refonologized-loan table → use the refonologized form (e.g., "waka", "kawallu", "lapis", "iskuyla", "sapatu", "asukar").
  3. Else (C is a recent / technical / foreign-only concept) → use the SPANISH surface form (e.g., "computadora", "televisor", "barco", "radio", "internet", "doctor", "Pedro").
```

**No tie-break dilemmas.** Tables L1b, L1c, L1d below enumerate every concept we've encountered. If a new concept appears at training time, classify with the decision tree above; if uncertain, default to step 3 (Spanish surface).

**`nisqa` marker policy (HARD RULE).** ALWAYS strip the calque marker `nisqa` (and orthographic variant `ñisqa`) when it immediately follows a Spanish-form loanword in narrative prose. The marker is pedagogically useful but absent from AmericasNLP 2021 test references; keeping it adds surface-form variation the model would have to learn to suppress.

- `Televisor nisqatam rantirqamusqa` → `Televisortam rantirqamusqa`
- `computadora nisqapi` → `computadorapi`
- `Tablet nisqawan` → `Tabletwan`

Suffixes that previously attached to `nisqa` re-attach directly to the loan stem.

**Proper nouns are NEVER refonologized.** `Pedro` stays `Pedro`, never `Pidru`. `Ecuador` stays `Ecuador`, never `Ikwadur`. `Kennedy` stays `Kennedy`. This is a hard rule that overrides L1e refonologization sound maps.

**Country/region/department names** (`Perú`, `Lima`, `Ayacucho`, `Apurímac`):
- In a Spanish sentence: keep the Spanish form.
- In a Quechua sentence: use the Quechua refonologization if the official MINEDU vocab gives one (`Piruw`, `Lima` → unchanged, `Aya Kuchu`, `Apu Rimaq`). Otherwise keep Spanish.
- These are listed explicitly in L2.

### L1 — Three loan categories (Manual §6, p. 79–82)

**(a) Avoid the loan if a Quechua word exists.** Recover the native term:

| Spanish loan (Chanka colloquial) | Quechua canonical |
|---|---|
| `hungu` (hongo) | `k'allampa` (Collao) / no native Chanka — leave |
| `mayur` (mayor) | `kuraq` |
| `mana susiguyuq` (inquieto) | `t'uki` (Collao) — leave |

**(b) Refonologized old loans** — write with the Quechua alphabet:

- `turu` (toro), `waka` (vaca), `kawallu` (caballo), `wutilla` (botella), `winu` (vino), `sapatu` (zapato), `uwiha` (oveja), `makina`, `chumpa`, `lapis`, `kasaka`, `kalsun`, `asukar`, `muchila`, `mansana`, `iskuyla` (escuela), `silular` (celular, alongside native `qayana`), `riru` (dedo), `siwara` (cebada), `sularu` (soldado), `hawun` (jabón), `sanawri` (zanahoria), `kuwarta` (cuarta), `isika` (esquina/-).

**Note on /b/, /d/, /f/, /g/ correspondences in refonologized loans (Sistema vocálico §3.1):**
- `/b/` → `[w]`: `wutilla` from "botella"
- `/d/` → `[r]`: `riru` from "dedo"
- `/f/` → `[ph]` → after R2 `[p]`
- `/g/` → `[w]` or `[y]`

These take Quechua suffixes normally: `Chay winu wutillata haywamuway`.

### L1b — Canonical category (b) table (CHANKA form)

This is the locked normalization output. All variant surfaces map TO these forms.

| Spanish source | Canonical Chanka | Common variant surfaces to normalize → canonical |
|---|---|---|
| vaca | `waka` | `vaca`, `wakas` |
| toro | `turu` | `toro` |
| caballo | `kawallu` | `kaballo`, `caballo` |
| zapato | `sapatu` | `zapatu`, `zapato` |
| oveja | `uwiha` | `uwija`, `oveja` |
| botella | `wutilla` | `botella`, `wotella` |
| vino | `winu` | `vino`, `winu` |
| lápiz | `lapis` | `lapiz`, `lápiz` |
| escuela | `iskuyla` | `escuela`, `iskuyla`, `eskuyla` |
| azúcar | `asukar` | `asúcar`, `azukar`, `azúcar` |
| jabón | `hawun` | `jabón`, `jawón`, `jawun`, `hajun` |
| manzana | `mansana` | `manzana`, `mansana` |
| machete / cuchillo | `kuchillu` | `cuchillo`, `kuchilo` |
| camisa | `kamisa` | `camisa` |
| pantalón | `pantalun` | `pantalón` |
| cebada | `siwara` | `cebada`, `ciwara` |
| zanahoria | `sanawri` | `zanahoria`, `sanaoria` |
| asno / burro | `asnu` | `asno`, `burro` (low priority) |
| cabra | `kawra` | `cabra`, `kabra` |
| gallina | `wallpa` | `gallina`, `wallpa` |
| Pedro (when refonologized) | `Pidru` | (proper noun — see L2; usually keep Spanish) |
| iglesia (building) | `wak'a` → Chanka `waka` (sacred); else keep `iglesia` if Christian context | `iglesia` |
| dedo | `riru` | `dedo` |
| cuarta (measure) | `kuwarta` | `cuarta` |

### L1c — Canonical category (c) table (SPANISH form, optional `nisqa`)

These are the recent/technical loans where Spanish surface is canonical. **The normalizer ALWAYS strips `nisqa`/`ñisqa`** when it follows one of these loans (see §L0 hard rule).

| Concept | Canonical surface | Variants to reject → canonical |
|---|---|---|
| computer | `computadora` | `kumputarura`, `kumputatura`, `computatora` |
| television | `televisor` | `tiliwisur`, `tilibisur`, `televisorm` |
| telephone | `teléfono` | `tilipunu`, `tilifunu`, `telefono` (no accent OK) |
| cell phone | `celular` | `silular`, `selular` |
| internet | `internet` | `internét`, `internit` |
| tablet | `tablet` | — |
| radio | `radio` | `raryu`, `radyu` |
| boat / ship | `barco` | `warku`, `bargu` |
| airplane | `avión` | `awyón`, `awyun` |
| missile | `misil` | `misal` |
| car | `carro` | `karru` |
| bus | `ómnibus` / `bus` | `umnibus` |
| computer-related (technology) | `tecnología` | `tiknulukiya` |
| government | `gobierno` | `gubirnu`, `kuwirnu` |
| ministry | `ministerio` | `ministiryu` |
| bank | `banco` | `wanku` |
| president | `presidente` | `prisidinti` |
| doctor (MD) | `doctor` | `ruktur` |
| nurse | `enfermera` | `inpirmira` |
| school day / class | `clase` | `klasi` |
| sister / brother | `hermana` / `hermano` | `irmana`, `irmanu` |
| COVID | `Covid` | — |
| Pedro (proper noun) | `Pedro` | `Pidru` (refonologized form rejected for proper nouns; see L2) |
| Ecuador | `Ecuador` | `Ikwadur` |

### L1d — Native Chanka preferred over loan (HARD MAPPING)

When EITHER the Spanish surface OR a refonologized loan appears in a Quechua sentence for a concept that has a native Chanka word, ALWAYS replace with the native Chanka word. Apply unconditionally — no context-dependent carve-outs.

| Spanish surface | Refonologized loan | → Canonical native Chanka |
|---|---|---|
| `mayor` | `mayur` | `kuraq` |
| `hablar` (in Quechua text) | `parlay` | `rimay` |
| `presidente` | `prisidinti` | `kamachikuq` |
| `país` | `payis` | `mama llaqta` |
| `región` / `departamento` | `riyun` | `suyu` |
| `hospital` | `uspital` | `hampina wasi` |
| `policía` | `pulisiya` | `wayruru` |
| `escuela` (institution sense) | `iskuyla` | `yachay wasi` |

**Note:** `iskuyla` (as a physical building) is the L1b canonical form per MINEDU. `yachay wasi` is the canonical form for the institution/concept. Disambiguator: if the surrounding text mentions students/learning, use `yachay wasi`; if it mentions a building/place, use `iskuyla`. *Edge case left to the normalizer's seq2seq context handling — both forms are acceptable in their respective sub-domains, but the L1d native form is the default.*

**`doctor` is a hard exception:** even though "médico" → `hampi kamayuq` is the native form, the contemporary register universally uses Spanish-form `doctor`. Canonical: `doctor`.

**`iglesia` is context-collapsing:** in modern usage `iglesia` (the building) is the canonical surface; the older calque `iñiy wasi` is not produced unless the source explicitly distinguishes Christian institution from sacred site. Canonical: `iglesia`.

### L1e — Refonologization sound map (for inverse normalization)

For a Spanish source word that someone wrote in refonologized form (and we want to convert it to canonical), or vice-versa:

| Spanish letter | Chanka mapping |
|---|---|
| `b` | `w` (`vaca → waka`, `botella → wutilla`) |
| `d` | `r` (`dedo → riru`, `radio → raryu`) |
| `e` | `i` (`escuela → iskuyla`, `febrero → piwriru`) |
| `f` | `p` in Chanka / `ph` in Collao (`física → isika`) |
| `g` | `w` or `k` (`agua → awa`, `Guayanas → Wayanas`) |
| `j` (Spanish /x/) | `h` (`jabón → hawun`, `julio → hulyu`) |
| `o` | `u` (`octubre → uktumri`, `lunes → lunis`) |
| `v` | `w` (`vino → winu`, `viernes → wirnis`) |
| `z` | `s` (`zapato → sapatu`, `azúcar → asukar`) |
| `c` (before `e/i`) | `s` (`ciudad → siwra`) |
| `c` (before `a/o/u`) | `k` (`camisa → kamisa`) |
| `qu` (before `e/i`) | `k` (`quilla → killa`) |
| `tr/pr/pl/bl` (consonant cluster) | broken by epenthetic vowel (`trompo → rumpu`, `plátano → latanu`) |

**(c) Recent loans (technology, abstractions)** — ALWAYS keep Spanish spelling; ALWAYS strip `nisqa`:

- `Televisor nisqatam rantirqamusqa.` ("(They) had bought a [thing called] television.")
- `Telefonoykita mañaykuway.` ("Lend me your telephone.")
- ⚠ The manual cautions: "no abusar de su uso" — don't over-apply `nisqa`. Use it for genuinely foreign technical concepts.

### L2 — Proper nouns of Spanish/foreign origin (HARD: keep Spanish spelling)

Keep Spanish spelling unchanged, attach Quechua suffixes directly:

- `Pedrochaqa ripunqam.` (Pedrito will leave)
- `Ecuadorpim llamkan.` (works in Ecuador)
- `Covidraykum, manam yachay wasiyman rinichu.` (because of Covid…)
- `Kennedy rimarqan Fuerza Aérea pilutukunawan.`
- `María kachkanmi.`

**NEVER refonologize a proper noun.** `Pedro` ↛ `Pidru`, `Ecuador` ↛ `Ikwadur`, `Kennedy` ↛ `Kinidi`, `Faulk` ↛ `Pawlk`, `José` ↛ `Husi`. This rule overrides §L1e.

### L3 — Anthroponyms / toponyms of Quechua origin (HARD: per-name table)

When the surrounding sentence is in Quechua, apply this fixed table:

| Spanish surface | Canonical Quechua form |
|---|---|
| Chosica | `Chusiqa` |
| Rimac (Lima river / district) | `Rimaq` |
| Huancavelica | `Wanka Willka` |
| Apurímac | `Apu Rimaq` |
| Machupicchu | `Machu Pikchu` |
| Cusco / Cuzco | `Qusqu` |
| Ayacucho | `Aya Kuchu` |
| Quispe (surname) | `Qispi` |
| Huamán (surname) | `Waman` |
| Andahuaylas | `Anta Waylla` |
| Cocachacra | `Kuka Chakra` |
| Titicaca | `Titi Qaqa` |
| Yanaoca | `Yana Uqa` |
| Vilcashuamán | `Willkas Waman` |
| Huancayo | `Wankayuq` |
| Azángaro | `Aswan Karu` |
| Choclococha | `Chuqllu Qucha` |
| Puno | `Punu` |
| Arequipa | `Ariq Qhipan` |
| Perú | `Piruw` |
| Lima | `Lima` (unchanged) |
| Ancash | `Anqas` |
| Junín | `Hunin` |
| Loreto | `Luritu` |
| Madre de Dios | `Madre de Dios` (unchanged) |
| Cajamarca | `Kasha Marka` |

When the surrounding sentence is in **Spanish** (e.g., the source side of a parallel pair), keep the standard Spanish spelling. The detector uses surrounding morphology (Quechua suffixes vs Spanish function words) to decide language.

---

## 6. Domain-specific cautions

### Bibles, religious corpora

| Source | Orthography | Action |
|---|---|---|
| **Chuya Qellqa** (Ayacucho 1987, rev. 2012) | ✅ MINEDU-trivocal Chanka | Use as-is, audit only |
| **Diospa Simin Qelqa** (Cusco 1988) | ❌ Pentavocalist (`Qelqa`, `qhepa`, `nacesqan`) | **EXCLUDE** from training (too many compound errors; cheaper to skip than fix) |
| **JW.org Cusco/Bolivia** | ❌ 5 vowels, `'`, `j`-for-`h` | **EXCLUDE** from training |
| **JW.org Ayacucho** | Unverified (`/qu-ay/` 404 at audit) | **INCLUDE** if confirmable as quy; run full normalizer |
| **JW300 `quy`** | Mixed lineage | **INCLUDE** every line; run full normalizer; do not pre-filter |

### Educational / MINEDU primary materials

- `Ñawinchana Maytu` (1° and 2° Primaria), `Willakuykuna Maytu`, `Yachachinapaq simikuna` (the vocab itself): MINEDU-canonical, use as-is.
- Older textbooks pre-2014 may use 5-vowel writing — apply full normalization.

### Wikipedia (qu.wikipedia.org)

The Quechua Wikipedia is mixed-variety. **EXCLUDE** pages tagged as Cuzco-Collao. **INCLUDE** pages tagged as Ayacucho-Chanka or untagged, after running the full normalizer.

### RunaSimi.de

**EXCLUDE** the lemma-fallback rows (16,231 with Cuzco-style orthography). **INCLUDE** the 3,339 high-confidence Ayakuchu rows after running the full normalizer.

### Nouman 2023 MT-SharedTask `extra/handbook_quy`

**EXCLUDE.** Labeled as `quy` but documented to contain `quz`/`quy` mixed orthography. Cost of normalizing exceeds the value of the data.

---

## 6.5 Numeral policy (HARD)

The MINEDU manual is silent on digit-vs-word. Decision: **keep numerals as written in the source**. Do not transcribe `20` ↔ `iskay chunka`. Both are acceptable; converting between them is a translation choice, not a normalization choice.

The only normalization on numerals: when a Quechua number-word is misspelled (`pisqa, pishqa → pichqa`), apply the wrong→right table.

## 6.6 Punctuation normalization (HARD)

- Smart quotes `"`, `"`, `'`, `'` → ASCII `"` and `'`. (Note `'` is otherwise forbidden in Chanka — apply only inside Spanish text spans or as quote delimiters.)
- Em dash `—` (U+2014) preserved for dialogue.
- Hyphen-minus `-` is for citation/morphology only; remove from running text.
- Spanish inverted `¿`, `¡` preserved.
- Multiple consecutive spaces collapse to single.
- Trailing whitespace stripped.

## 7. The wrong → right gold list (test set for normalizer)

This list serves both as ground truth for unit tests and as a seed for any future supervised normalization pass.

### 7.1 Apostrophe stripping (Rule R1)

```
p'acha → pacha
t'anta → tanta
mayt'u → maytu
q'illu → qillu
k'aspi → kaspi
ch'uklla → chuklla
sach'a → sacha
wat'a → wata
hap'iy → hapiy
q'apiy → qapiy
mut'i → muti
wik'uña → wikuña
llamk'ay → llamkay
mink'a → minka
siq'uy → siquy
wirp'a → wirpa
samp'a → sampa
p'inqay → pinqay
p'uchqu → puchqu
hayk'ap → haykap
huk'ucha → ukucha
wichq'ay → wichqay
hump'i → humpi
k'umu → kumu
k'ispiñu → kispiñu
hak'akllu → akakllu
mach'ay → machay
q'iwiy → qiwiy
hich'ay → hichay
wayq'u → wayqu
surq'an → surqan
hak'u → aku
huch'uy → uchuy
t'ika → tika
t'ampa → tampa
mat'iy → matiy
q'aytu → qaytu
q'umir → qumir
hasp'iy → aspiy
rikch'aq → rikchaq
tiqt'iy → tiqtiy
```

### 7.2 De-aspiration (Rule R2)

```
phullu → pullu
thuqay → tuqay
chhalla → chala
qhari → qari
mikhuy → mikuy
khuchi → kuchi
phukuy → pukuy
phatma → patma
thuñiy → tuñiy
llanthu → llantu
qhincha → qincha
qhaway → qaway
ichhu → ichu
aqha → aqa
ñaqha → ñaqa
raphi → rapi
saphi → sapi
qhachun → qachun
khuyapayay → kuyapayay
uphakuy → upakuy
walthay → waltay
```

### 7.3 Five-to-three vowel (Rule R3)

```
ñoqa → ñuqa
noqa → ñuqa
qelqay → qillqay
qellqay → qillqay
qosqo → qusqu
qhepa → qipa
sonqo → sunqu
qollqe → qullqi
qolqe → qullqi
qello → qillu
q'ello → qillu
onqoy → unquy
tukoy → tukuy
lloqsiy → lluqsiy
qopa → qupa
qom → qum
qoča → qucha
oqa → uqa
urqo → urqu
qechwa → qichwa
hampeq → hampiq
onqoq → unquq
unqox → unquq
```

### 7.4 Forbidden grapheme replacement (Rule R4)

```
jam → qam
jucha → hucha
jampi → hampi
qampi → hampi
jawapi → hawapi
qawapi → hawapi
jatun → hatun
qatun → hatun
atun → hatun
quilla → killa
camay → kamay
ccapacc → qapaq
ccollque → qullqi
fukuy → pukuy
chhafchiy → chapchiy
huj → huk
huq → huk
mihuna → mikuna
mijuna → mikuna
pishqa → pichqa
pisqa → pichqa
phisqa → pichqa
ashka → achka
askha → achka
washka → waska
wachka → waska
kuchka → kuska
kushka → kuska
ch'uqay → chuqay
chhuqay → chuqay
wisq'ay → wichqay
wishqay → wichqay
machkay → maskay
mashkay → maskay
killi → qilli
khilli → qilli
qhillu → qillu
k'illu → qillu
killu → qillu
hata → qata
khata → qata
```

### 7.5 Diphthong breaking (Rule R5)

```
huaita → wayta
maitu → maytu
huauqei → wawqiy
yahuar → yawar
huata → wata
uata → wata
huaina → wayna
ccatimuhuay → qatimuway
wasy → wasi
misky → miski
```

### 7.6 `<l>` vs `<ll>` near `q` (Rule R6)

```
qulqi → qullqi
walqa → wallqa
wallka → wallqa
salqa → sallqa
salq'a → sallqa
alqu → allqu
allku → allqu
alqo → allqu
lliqlla → lliklla
llihlla → lliklla
```

### 7.7 `<m>`/`<n>` (Rule R7)

```
qan → qam
allim → allin
wasimpi → wasinpi
t'anpa → tampa
tanpa → tampa
kinsa → kimsa
```

### 7.8 Other regularizations

```
chiqaq → chiqap
llamphu → llampu
haykaq → haykap
hayk'aq → haykap
usqaylla → utqaylla
uthqaylla → utqaylla
qasqi → qatqi
llant'u → llantu
ruqhu → ruqu
luqu → ruqu
laphi → rapi
rap'i → rapi
ukyay → upyay
upyay → upyay
ufyay → upyay
uhay → upyay
ruway → ruray
mayqin → mayqan
hallp'a → allpa
hup'a → upa
opa → upa
llapan → llapa
llipin → llapa
tuyuy → tuytuy
warmiq → warmipa (Chanka genitive)
wasiki → wasiyki
mamanchis → mamanchik
mamanchix → mamanchik
ñuqapuwan → ñuqantin (use -ntin)
ñuqapiwan → ñuqantin
```

---

## 7.9 Reasoning-trace policy (HARD)

The student normalizer model (Qwen3.5-4B) is trained to emit a `<think>...</think>` block before the normalized text. The trace lists per-token decisions, each citing the spec rule (§R1–§R7, §S1–§S8, §L0–§L3, §6.5–§6.6) that justifies it.

**Trace format (locked):**

```
<think>
Tokens:
1. "<token>" — <observation>. Spec §<rule_id>: <rule_summary>. Result: "<normalized_token>"
2. ...
N. "<token>" — already MINEDU-compliant.
</think>

Normalized: <output sentence>
```

**Rules:**

- Every token receives an entry, even if no change is needed (write "already MINEDU-compliant").
- Cite ONE spec rule per change. If multiple rules apply, cite the highest-priority one (R-rules > S-rules > L-rules > §6 rules).
- Rule summaries must be ≤12 words. Example: "Spec §R1: strip Collao ejective marker".
- The trace MUST NOT propose alternatives or hedge ("could be", "maybe"). Each cited rule produces a determined output.
- The trace lists tokens left-to-right in source order.
- Empty trace bodies are NEVER allowed; if no token needs normalization, the trace says "All tokens already MINEDU-compliant. No changes required."

**Why traces matter:**

1. Linguists can audit each decision against the spec.
2. Catastrophic failures (hallucinations) become visible: a trace that fails to mention a clearly Cuzco-form token is a debug signal.
3. The model generalizes from reasoning about rules, not from surface memorization of mappings — better OOD behavior on previously-unseen wrong-form variants.

## 8. Operational pseudocode for the normalizer

```
def normalize_chanka(text: str) -> NormResult:
    # Tokenize keeping Spanish spans intact (proper nouns, recent loans).
    spans = segment_quechua_vs_spanish(text)
    out = []
    flags = []
    for span in spans:
        if span.kind == "spanish_span":
            out.append(span.text)
            continue
        # Quechua span — apply rules in order.
        t = span.text
        t = strip_apostrophes_outside_words(t)        # R1 (also kills standalone ' that is not contracted form)
        t = deaspirate(t)                              # R2:  chh|kh|ph|qh|th -> ch|k|p|q|t
        t = deglottalize(t)                            # R1 in-word: ch'|k'|p'|q'|t' -> ch|k|p|q|t
        t = enforce_three_vowels(t)                    # R3: e→i, o→u (Quechua tokens only)
        t = replace_forbidden_graphemes(t)             # R4: j→{q,h,k}, cc/qu/c→k, f→p, b→w, d→r
        t = break_diphthongs(t)                        # R5
        t = ll_before_q(t)                             # R6
        t = m_n_before_p(t)                            # R7
        t = preserve_full_suffix_forms(t)              # S2: -ki → -yki, -nchis → -nchik
        # Validate
        if violates_chanka_inventory(t):
            flags.append(("invalid_grapheme_remaining", t))
        out.append(t)
    return NormResult(text="".join(out), flags=flags)
```

Tests assert that every example in §7 maps exactly. Failure on any test blocks the build.

---

## 8.5 Conservative fallback (HARD)

If a token does NOT match any of the following:
- A spec rule from §R1–§R7 / §S1–§S8 / §L1d
- An entry in the §7 wrong→right gold list
- An entry in the §L1b / §L1c / §L1d / §L3 canonical tables

then PRESERVE the token as-is. Do not invent a normalization.

This rule prevents teacher LLMs and the student model from hallucinating mappings for OOV tokens. The training data will be biased toward "no change" when uncertain — exactly what we want.

**Examples of preserve-as-is:**

- Religious proper nouns: `Dios`, `Jesús`, `Cristo`, `Espíritu Santo`, `Señor` (keep Spanish per §L2).
- Acronyms: `MINEDU`, `EIB`, `UNSAAC`, `DRE`, `DRELM`, `INEI`.
- URLs, emails, hashtags, dates (`1985-11-18`), times (`9:30`).
- Onomatopoeia: `pah pah`, `wak wak` (any all-lowercase repeated CVC).
- Unknown OOV Quechua-shaped tokens that don't match any rule.

## 8.6 Multi-word Spanish spans inside Quechua sentences (HARD)

If 3+ consecutive tokens are all Spanish-shaped (Spanish function words `de`, `la`, `que`, `con` etc. with no Quechua suffixes), treat the entire span as a Spanish quote and preserve unchanged. Examples:

- `Pay nirqa "muy bien gracias" nispa.` → preserved
- `Atuqsi nirqa "Dios mío" nispa.` → preserved

Single Spanish words attached to Quechua suffixes (`Ecuadorpim`, `televisortam`) are NOT spans — they go through the normal loan rules.

## 9. Out-of-scope (do NOT normalize)

1. Spanish text on the source side of parallel data.
2. Spanish proper nouns (`Pedro`, `María`, `Kennedy`, `Ecuador`, `Covid`).
3. Refonologized loans whose canonical Chanka form already uses non-Quechua-looking spelling (`asukar`, `borraja` listed as Spanish-kept in the vocab).
4. Quoted Spanish inside Quechua text.
5. Digits — the MINEDU manual is silent on digit-vs-word; do not coerce.

The segmenter (step 1 of §8) is responsible for detecting these spans. False positives (treating Spanish as Quechua) are the dominant failure mode and must be guarded by a Spanish-token detector.

---

## 10. Validation plan

### 10.1 Unit tests
Every wrong→right pair in §7 (~120 cases) must map exactly. CI blocks on regression.

### 10.2 AmericasNLP 2021 test compliance audit
Profile the 1003 quy reference lines:
- `% of lines containing { ', e, o, j, c (not in ch), cc, qu, f, b, d, g, v, x, z }`
- `% of lines containing any of { chh, kh, ph, qh, th, ch', k', p', q', t' }`

Expected: >95% compliance (the test set is post-MINEDU). If any line *fails*, examine — there may be edge cases the spec doesn't cover.

### 10.3 Round-trip on Chuya Qellqa
Apply normalizer to a sample of Chuya Qellqa (which is already MINEDU-compliant). Output should equal input. Any change is a normalizer bug.

### 10.4 Effective change measurement on JW300/Wikipedia
Apply normalizer to JW300 quy and Quechua Wikipedia samples. Measure:
- avg edit distance per sentence
- top-100 most-frequent normalization replacements (sanity check)

### 10.5 Final acceptance
Train v34 (or successor) on normalized data and measure delta on AmericasNLP 2021 test ChrF vs v30 baseline.

---

## 11. Citations

- Cerrón-Palomino, R. (1987). *Lingüística Quechua*. Cusco: Centro Bartolomé de las Casas.
- Cerrón-Palomino, R. (1994). *Quechua sureño: diccionario unificado*. Lima: Biblioteca Nacional del Perú.
- Chuquimamani Valer, N. R., Chávez Gonzales, O., Riveros Paravicino, F. A., Jara Luna, C., Cárdenas Guzmán, M., Quintasi Mamani, M. (2021). *Urin Qichwa Qillqay Yachana Mayt'u / Manual de escritura quechua sureño*. Lima: MINEDU-DEIB. [PDF](https://repositorio.minedu.gob.pe/handle/20.500.12799/7190)
- Dhawan, A., Driggers-Ellis, C., Grant, C., Wang, D. (2026). Improving Indigenous Language Machine Translation with Synthetic Data and Language-Specific Preprocessing. *Proceedings of LoResMT 2026*. [aclanthology.org/2026.loresmt-1.10](https://aclanthology.org/2026.loresmt-1.10.pdf)
- Gow-Smith, E., Sánchez Villegas, D. (2023). Sheffield's submission to the AmericasNLP shared task on machine translation into indigenous languages.
- Helsinki-NLP team (2021). The Helsinki submission to the AmericasNLP shared task. [aclanthology.org/2021.americasnlp-1.29](https://aclanthology.org/2021.americasnlp-1.29.pdf)
- MINEDU (1985). *Resolución Ministerial N° 1218-85-ED*: official Quechua and Aymara alphabets.
- MINEDU (2021). *Yachachinapaq simikuna / Vocabulario pedagógico quechua sureño*. Lima: MINEDU-DEIB. [PDF](https://repositorio.minedu.gob.pe/bitstream/handle/20.500.12799/7490/Yachachinapaq%20simikuna%20-%20Urin%20Qichwa%20vocabulario%20pedag%C3%B3gico%20quechua%20sure%C3%B1o.pdf)
- Pantigozo Montes (2022). *El sistema vocálico del quechua sureño*. MINEDU-DEIB. [PDF](https://repositorio.minedu.gob.pe/handle/20.500.12799/10382)
- Soto Ruiz, C. (1976). *Gramática Quechua: Ayacucho-Chanca*. Lima: IEP / MINEDU.
- Soto Ruiz, C. (2012). *Runasimi-Kastillanu-Inlis Llamkaymanaq Qullqa: Ayakuchu-Chanka / Diccionario funcional quechua-castellano-inglés*. Volumes I–III.
