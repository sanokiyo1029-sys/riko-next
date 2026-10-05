# Voices

`riko_sample.wav` is the sample reference clip shipped with the original
[Project Riko](https://github.com/rayenfeng/riko_project) (`character_files/main_sample.wav`,
MIT License, by Rayen Feng). Its transcript is the `prompt_text` in `config.example.yaml`.

GPT-SoVITS reference clips must be **3-10 seconds**, clean, without long silences, and
`prompt_text` must be the exact words spoken in the clip. For emotion-aware speech, record
one clip per emotion (e.g. `riko_happy.wav`) and map them under `tts.voices.<name>.emotions`.
