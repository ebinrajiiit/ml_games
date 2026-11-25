# Play the model: QIPPG103 classroom games

Three phone games for Dr. Sara Renjit's sessions in **QIPPG103 Machine Learning: Principles and Practices** at IIIT Kottayam (Lectures 20, 21 and 24).
Static pages, same set-up as [`phd_games`](https://github.com/ebinrajiiit/phd_games). Only a player's nickname and score are sent to the room screen.

| Game | File | Lecture | What it does |
| --- | --- | --- | --- |
| Hidden State | `hidden-state.html` | 20, RNN | Tune `w_hh` and `w_xh` of a one-number RNN to match five memory curves; keep a gradient alive for 40 steps; four quick questions on RNN shapes. |
| Gatekeeper | `gatekeeper.html` | 21, LSTM/GRU | Set the forget, input and output gates of a one-slot LSTM, step by step: store, overwrite, count, recall twice; then one level as a GRU. |
| Reward Designer | `reward-designer.html` | 24, RL | Four task-formulation cards, then three broken reward designs for a gridworld delivery robot. Training is exact value iteration, so results are reproducible. |

Each game scores out of 100. Live at <https://ebinrajiiit.github.io/ml_games/>. The QR codes in the slides point to `https://ebinrajiiit.github.io/ml_games/<file>`; if the repository is ever renamed, change `\gameurl` in `slides/common.tex`.

## Room screen

`results.html` is the projector view (live player count, average, histogram, top five, and a per-round breakdown you reveal at the end). Open it as `results.html?game=rnn`, `?game=gate` or `?game=mdp`. Add `&session=qip-L21` to show a particular class without editing `config.js`.

## Host setup

1. **Supabase.** This uses the same project and `plays` table as `phd_games`. Run `supabase.sql` once in the SQL Editor: it widens the table's game check to allow `rnn`, `gate` and `mdp`. Nothing else changes.
2. **Session.** Set `SESSION` in `config.js` before each lecture (for example `qip-L20`) so the room screen starts empty, or pass `?session=` to the room screen.
3. **GitHub Pages.** Settings → Pages → Deploy from branch → `main`, root.

The games still work without Supabase; they just don't report to the room screen.

## Lecture materials

- `slides/`: Beamer source and PDFs for Lectures 20, 21 and 24. Compile with `pdflatex` (twice). Lists and diagrams build step by step; each PDF page is one click.
- `worksheets/`: student copies (PDF). The LaTeX sources and answer keys are kept off this public repository.

## Case-study code (`case_study/`)

| File | Lecture | Needs |
| --- | --- | --- |
| `char_rnn_places.py` + `kerala_places.txt` | 20 | NumPy. A character-level RNN with hand-written BPTT that invents Kerala place names. Under a minute on a laptop. |
| `memory_length_experiment.py` | 21 | JAX (`pip install jax`). Plain RNN vs LSTM vs GRU on remembering the first symbol for T steps. About 10 minutes on a CPU. |
| `campus_courier_env.py` | 24 | `gymnasium`, NumPy. The Campus Courier environment, a random-policy baseline and two hand-written routes. Passes `gymnasium`'s `check_env`. |

Fonts are self-hosted from `fonts/` (Public Sans and Source Serif 4, SIL Open Font License).
