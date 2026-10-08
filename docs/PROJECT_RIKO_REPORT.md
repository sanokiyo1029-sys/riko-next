# Project Riko - research report (text copy)

Source: `PROJECT_RIKO_REPORT.pdf` (Oct 8, 2026). Extracted text below; the PDF is authoritative for tables.
Riko's `knowledge` list in `characters/riko.yaml` and the VMC output (`server/riko/vmc.py`) are based on it.

```text
Project Research Report: Riko Project

 Riko Research Report: The Whole Project

   Oct 8, 2026 · ​@Dread

 1. Executive Summary

  Riko is three things: a small public voice-chat repo, a VTuber avatar layer shown only on
  video, and a paid app on a waitlist. Only the first can be checked in code, and its README
  overstates it.
  Method and limits. For the repo, I cloned rayenfeng/riko_project and read every file, its 5
  commits and all 28 issues and pull requests. I did not run it. For the avatar layer, I used the
  video research report on "My AI Waifu is A VTuber Now": 310 frames with OCR plus
  YouTube's auto-captions, since the file had no audio. For Rikoverse, I read its public
  landing page. I checked versions against GitHub, PyPI and OpenAI's docs.
  Layer 1, the public repo: works, but the README is wrong in places. It is about 200 lines
  of Python: push-to-talk, Faster-Whisper, OpenAI gpt-4.1-mini , then a GPT-SoVITS
  server. The flow matches the code. But the config file name is wrong, the run command
  fails as written, the GPU steps do nothing (the code is fixed to CPU), and the bundled
  config points to the author's own D:\ drive. It says "MIT" with no LICENSE file. Nothing
  has been committed since 2025-06-24.
  Layer 2, the avatar: real technique, no public code. The video drives a VRoid model in
  VSeeFace with VMC/OSC messages from Python, and the calls it settles on match the
  VMC spec. ChatGPT's suggested /VMC/Ext/Blend/Expression address does not exist,
  and the video shows it failing. The motion is scripted idle and talking loops, not AI-driven.
  None of this code is in the repo.
  Mismatched model story. The repo uses OpenAI's paid API. The video credits a local
  model, TheBloke/Ana-v1-m7-GGUF, which is licensed for non-commercial use only. The
  repo has no local-LLM option.
  Layer 3, Rikoverse: a $7/month waitlist. The video's "secret link" leads to it. It promises
  memory and live avatar reactions; none of it can be checked. The repo, with 1,155 stars and
  215 forks, never mentions it.
  Outdated pieces. Python 3.10 (the tested version) reached end of life on 2026-10-01. The
  VSeeFace build shown lacks a Unity security patch. Claude 3.7 Sonnet, which wrote the
  avatar animations, was retired on 2026-02-19.

                                                                                                                                                                                                                                                    Page 1 of 13

Project Research Report: Riko Project

2. What Riko Is

Riko is an anime-styled "snarky" AI companion that listens, answers in a cloned voice and,
on video, moves as a VTuber avatar.
Creator. "Just Rayen" (rayenfeng on GitHub), whose profile reads "Trying to become data
scientist." He documents the build on YouTube and sells early access through Rikoverse.

The three layers:

Layer    What it does                  Where it lives        Can it be
                                                             checked?
Voice    Mic → Faster-Whisper →        Public repo, server/  Yes, source read in
loop     OpenAI gpt-4.1-mini → GPT-                          full
Avatar   SoVITS → speakers, with chat  Videos only
         memory                                              Partly, from on-
Product                                rikoverse.com,        screen code
         VRoid model in VSeeFace,      $7/month waitlist
         moved by VMC/OSC messages                           No, claims only
         from Python; lip sync from
         audio via VB-CABLE

         Private chats, memory, "live
         avatar reactions", evolving
         personality

                                                             Page 2 of 13

Project Research Report: Riko Project

Riko's three layers · what is public, video-only and paid

The GitHub repo stops at the speakers. Everything that makes Riko a VTuber sits in the
dashed layers, which you can only see on video or buy.

History. The repo was created and fully committed on 2025-06-24 (5 commits). Its
description calls it "The public version of the Riko Project", implying a fuller private one.
The VTuber video dates to 2025, going by the models it shows.

Files in the repo:

File                                         What it does

server/main_chat.py                          The loop: record, transcribe, ask the
server/process/asr_func/asr_push_to_talk.py  LLM, speak, delete temp audio, repeat

server/process/llm_funcs/llm_scr.py          Records the mic between two Enter
                                             presses, then transcribes with
                                             Whisper base.en

                                             Calls OpenAI's Responses API and
                                             stores the whole chat in

                                             chat_history.json

                                             Page 3 of 13

Project Research Report: Riko Project              What it does

    File                                           POSTs the reply to GPT-SoVITS on
                                                   127.0.0.1:9880 and plays the WAV
     server/process/tts_func/sovits_ping.py        API key, model name, system prompt,
     character_config.yaml                         voice settings
     character_files/main_sample.wav
     client/still_in_development.txt               The reference voice clip (735 KB)

                                                   Empty placeholder

3. Key Claims

Of 27 claims checked across the repo, the video and Rikoverse, 7 are verified, 7 mostly
accurate, 7 misleading, 3 false and 3 cannot be verified.

#  Layer  Where    Claim                           Verdict      Evidence

1  Repo   README   "She listens, and remembers     Mostly       Full history is saved to

          intro    your conversations"             accurate     chat_history.json and re-sent

                                                                every turn. It is never trimmed.

2  Repo   README   A "fully configurable           Misleading   Only prompt, model and voice are in
                                                                config. Whisper model, CPU device,
          intro    conversational pipeline"                     server URL and sample rate are hard-
                                                                coded.

3  Repo   README,  "All prompts and parameters are False        The file is character_config.yaml .
                                                                No config.yaml exists.
          Configuration stored in config.yaml "

4  Repo   README, top "tested with python 3.10     Cannot be No tests or CI. Python 3.10 reached

                   Windows >10 and Linux Ubuntu" verified       end of life on 2026-10-01.

5  Repo   README,  Install with uv pip install -r Mostly        Installs, but requirements.txt is

          Setup    extra-req.txt then              accurate mostly GPT-SoVITS's dependencies.

                   requirements.txt                             install_reqs.sh uses steps the

                                                                README never mentions.

6  Repo   README,  GPU support for Faster-Whisper Misleading main_chat.py hard-codes

          Setup    needs CUDA and cuDNN                         device="cpu" .

7  Repo   README,  "Launch the GPT-SoVITS API"     Mostly       The /tts call on port 9880 matches

          Usage                                    accurate GPT-SoVITS's api_v2.py . No launch

                                                                steps given.

8  Repo   README,  Run python main_chat.py         False        The script is server/main_chat.py . It
                                                                works only as python
          Usage                                                 server/main_chat.py from the repo
                                                                root.

9  Repo   README,  Six-step flow: listen, transcribe, Verified  Matches main_chat.py line for line.

          Usage    GPT, respond, synthesize, play

                                                                                                  Page 4 of 13

Project Research Report: Riko Project

#   Layer  Where                       Claim                            Verdict     Evidence

10  Repo   README,                     GPT-SoVITS, Faster-Whisper,      Verified    All three are used. The first two are

           Credits                     OpenAI GPT                                   MIT licensed.

11  Repo   README,                     "MIT — feel free to clone,       Misleading  No LICENSE file. GitHub reports no
                                                                                    license.
           License                     modify"

12  Repo   README,                     GUI, live mic, emotion control   Verified    None exist. No commits since 2025-

           TODO                        and VRM frontend are future                  06-24.

                                       work

13  Repo   Description "The public version of the Riko Misleading Implies a usable cut of what the

                                       Project"                                     videos show. No avatar, client or local-

                                                                                    LLM code.

14  Repo   Config default Works with the bundled voice                  Misleading ref_audio_path points to

                                       sample                                       D:\PyProjects\waifu_project\... ,

                                                                                    the author's own drive.

15  Video  Title, 0:00                 "My AI Waifu is a VTuber now"    Mostly      Motion is scripted idle and talking

                                                                        accurate loops plus audio lip sync, not driven by

                                                                                    the LLM.

16  Video  2:02                        ChatGPT: VMC beats "OSC          False       No such address in the VMC spec,
                                       Expressions"                                 and VMC itself runs over OSC. The
                                       ( /VMC/Ext/Blend/Expression )                video shows it failing (2:35–3:32).

17  Video  ~4:02                       VSeeFace's VMC receiver          Verified    VMC spec and VSeeFace docs.
                                       defaults to port 39539

18  Video  4:21–4:47                   Blink via Blend/Val then         Verified    Matches the spec. Names are case-
                                       Blend/Apply                                  sensitive.

19  Video  4:50–5:10                   Bone/Pos numbers 4–6 are nod,    Mostly      They are quaternion x, y, z, which act
                                       twist and side-bend              accurate    like pitch, yaw, roll only for small
                                                                                    angles.

20  Video  ~4:13                       Joints are sent "every           Mostly      Spec says every update cycle. "Every
                                       millisecond or something"        accurate    millisecond" is hyperbole.

21  Video  5:11–7:22                   Commands run slowly unless       Cannot be   Cause not shown. VSeeFace's
                                       sent twice                       verified    smoothing slider is one guess.

22  Video  ~8:05–8:17                  "LLM leaderboards" show          Misleading The OpenRouter page ranks by token

                                       Claude is the better programmer              usage, not quality.

23  Video  8:18–8:58                   The final animations came from   Verified    On-screen chat and notebook agree.
                                       Claude 3.7 Sonnet

24  Video  ~9:00                       GPT-SoVITS → WAV → VB-           Verified    Consistent with each tool's docs.
                                       CABLE → VSeeFace

25  Video  Credits                     Local LLM is TheBloke/Ana-v1-    Mostly      The model exists, but is CC-BY-NC-
                                       m7-GGUF                          accurate    4.0, and the repo uses OpenAI
                                                                                    instead.

26  Rikoverse Video                    "Secret link?? not virus"        Misleading  It is a paid early-access waitlist
                                                                                    ( utm_campaign=alpha ).
           description

27  Rikoverse Landing page Memory, "live avatar reactions", Cannot be Not launched. "Access opens soon";

                                       evolving personality, $7/month verified      nothing to test.

                                                                                                                        Page 5 of 13

Project Research Report: Riko Project

 4. Detailed Fact Check

  Each item separates what Riko's README or video says from what the code and
  independent sources show. Links are in section 7.

 A. Repo and setup

  A1. Config file and run command: False. The README says to edit config.yaml and run
   python main_chat.py . The real file is character_config.yaml , and the script is in
   server/ . Both llm_scr.py and sovits_ping.py open character_config.yaml relative
  to the current folder. So the only working command is python server/main_chat.py
  from the repo root. Issue #21 ("files not found running the setup") is a likely result.
  A2. Dependencies: Mostly accurate. extra-req.txt holds what Riko's own code uses:
   faster-whisper , sounddevice , openai . requirements.txt is about 45 packages,
  including Chinese, Japanese and Korean text tools ( funasr , pypinyin , pyopenjtalk ,
   g2pk2 ). Riko's code imports none of them; they match GPT-SoVITS's needs.
   install_reqs.sh takes a different path: CUDA 12.6 PyTorch 2.6.0, --no-deps , and NLTK
  downloads. The README never mentions it. llm_scr.py imports gradio but never uses
  it.
  A3. GPU support: Misleading. The README lists CUDA and cuDNN "for Faster-Whisper
  GPU support". main_chat.py loads WhisperModel("base.en", device="cpu",
  compute_type="float32") . A GPU only helps GPT-SoVITS, which runs separately. Issue
  #24 asks about AMD GPUs; no maintainer answered.
  A4. Voice sample path: Misleading. ref_audio_path is
   D:\PyProjects\waifu_project\riko_project\character_files\main_sample.wav .
  GPT-SoVITS reads that path itself, so anyone else must edit it first. The README does not
  say so.
  A5. License: Misleading. The README says "MIT", but the repo has no LICENSE file and
  GitHub's API reports license: null . The intent is clear, but without the license text,
  reuse terms are informal. Who owns the voice in main_sample.wav is not stated.
  A6. Maintenance: Verified as stale. All 5 commits are from 2025-06-24. 22 issues and PRs
  are open, including fixes (#12 "Fix crash on tts failure") and Ollama support (#16). The
  owner has not commented on any issue or merged any outside PR.

                                                                                                                                                                                                                                                    Page 6 of 13

Project Research Report: Riko Project

 B. Voice and LLM pipeline

  B1. Flow: Verified. main_chat.py runs exactly the README's six steps, then deletes
  every .wav in audio/ .
  B2. Memory: Mostly accurate. Every turn is appended to chat_history.json and the
  whole file is sent each time. gpt-4.1-mini has a context window of about 1M tokens, so it
  will hold a long time. But input cost grows every turn, and nothing summarizes or trims.
  B3. Push-to-talk limits: Not mentioned. Recording runs between two Enter presses and is
  capped at 60 seconds ( sd.rec(int(60 * samplerate)) ). Longer speech is cut silently.
  B4. GPT-SoVITS call: Verified. POST http://127.0.0.1:9880/tts with text ,
   text_lang , ref_audio_path , prompt_text , prompt_lang matches GPT-SoVITS's
   api_v2.py , whose default port is 9880. If the server is down, sovits_gen returns None
  but the loop still tries to play the file and crashes. A comment mentions a "START ALL
  SERVER SCRIPT" that is not in the repo.
  B5. Model mismatch: Mostly accurate. The video credits TheBloke/Ana-v1-m7-GGUF, a
  local model under CC-BY-NC-4.0. The repo only calls OpenAI. Five issues (#4, #9, #13, #15,
  #20) ask for a free or local option.

 C. Avatar control (VMC/OSC, video only)

  C1. ChatGPT's "OSC Expressions": False. VMC is built on OSC. It defines
   /VMC/Ext/Blend/Val (name and value) and /VMC/Ext/Blend/Apply . There is no
   /VMC/Ext/Blend/Expression . The video honestly shows it failing (2:35–3:32).
  C2. Blink and expressions: Verified. The Blend/Val + Blend/Apply pattern is what the
  spec describes. "Blink" is a VRM0 preset name; names are case-sensitive.
  C3. Decoding Bone/Pos : Mostly accurate. The spec format is name, px, py, pz, qx, qy, qz,
  qw: a position, then a quaternion. Numbers 4–6 act like nod, twist and side-bend only for
  small angles, and the 7th (qw) is needed too.
  C4. "Send twice" and "every millisecond": Cannot be verified / hyperbole. The spec says
  bones are sent each update cycle, at a rate set by the sender. The slow-command
  workaround is unexplained.
  C5. Audio pipeline: Verified. GPT-SoVITS writes a WAV, VB-CABLE forwards it, and
  VSeeFace lip-syncs to that audio input. VB-CABLE is donationware; its commercial terms
  are unverified.

                                                                                                                                                                                                                                                    Page 7 of 13

Project Research Report: Riko Project

 D. AI coding claims

  D1. "Leaderboards say Claude is the better programmer": Misleading. The OpenRouter
  page shown ranks by token usage, not quality. Anthropic did claim state-of-the-art on
  SWE-bench Verified for Claude 3.7 Sonnet (2025-02-24), but the video does not cite it.
  That page now returns a 404.
  D2. "The animations came from Claude": Verified as a statement of what he did, not of
  which model is better. Claude 3.7 Sonnet was retired on 2026-02-19.

 E. Licensing and commercial

  E1. The "secret link": Misleading. It goes to rikoverse.com with a campaign tag. The page is
  a limited-alpha waitlist: "Early Access $7/month", "No charge today", "Access opens soon".
  The sales motive is not said on camera.
  E2. What Rikoverse promises: Cannot be verified. Private chats, memory, live avatar
  reactions and an evolving personality. The public repo has the memory part only, and no
  avatar code. The site does not mention open source or GitHub.
  E3. Credited model license: Mostly accurate. Ana-v1-m7 is non-commercial. Whether the
  paid service uses it is unknown.

 5. Important Information Riko Leaves Out

  The biggest gaps are the missing avatar code, the API-key and network risks, and the real
  cost.
   1. The avatar code is not public. The VMC/OSC scripts, notebooks and animations from

      the video are not in the repo. Issue #10 asks how the avatar works; only other users
      answered.
   2. API key in a tracked file. The OpenAI key goes in plain text in character_config.yaml ,
      which is committed, and there is no .gitignore . Pushing a fork can leak the key.
       chat_history.json (every conversation) is not ignored either.
   3. Listener bound to every interface. In the video, the OSC listener binds to 0.0.0.0 .
      Anyone copying it should use 127.0.0.1 .
   4. Ongoing cost. OpenAI is paid per token: gpt-4.1-mini is $0.40 per 1M input and $1.60
      per 1M output tokens. Because the full history is re-sent every turn, each reply costs
      more than the last. Rikoverse adds $7/month.

                                                                                                                                                                                                                                                    Page 8 of 13

Project Research Report: Riko Project

 5. Hardware. GPT-SoVITS is built around NVIDIA CUDA; install_reqs.sh pins CUDA
     12.6. VSeeFace is Windows-only and supports VRM0 models only.

 6. No error handling. If GPT-SoVITS is down, the loop crashes. If the OpenAI call fails, the
     turn is lost.

 7. Port setup. Neither the repo nor the video shows GPT-SoVITS's launch command or
     VSeeFace's send and receive ports. VSeeFace recommends different ports for each so
     it does not talk to itself.

 8. Licensing. No LICENSE file, a non-commercial model credited in the video, unstated
     rights to the sample voice, and unverified VB-CABLE terms.

 9. Simpler routes. The mouth can be driven with the A/I/U/E/O blend shapes over
      Blend/Val , with no virtual cable. @pixiv/three-vrm renders VRM in a browser with no
     VSeeFace.

10. Check LLM-written code. The video's own lesson: ChatGPT's first OSC address did
     not exist. Check generated API calls against the spec.

6. Current Status

As of 2026-10-08, the Python version, the VSeeFace build, the Claude model and the
OpenRouter page used for Riko have all been superseded. The repo has not changed
since 2025-06-24.

Item                                   In Riko                    Now
Repo activity                          5 commits, all 2025-06-24
                                                                  No commits since. 1,155
Python                                 "tested with python 3.10"  stars, 215 forks, 22 open
OpenAI model                                                      issues and PRs, no owner
                                       gpt-4.1-mini               replies
openai package
                                       Unpinned                   3.10 reached end of life on
                                                                  2026-10-01 (final release
                                                                  3.10.22)

                                                                  Not on OpenAI's
                                                                  deprecations page. gpt-
                                                                  4.1-nano shuts down 2026-
                                                                  10-23

                                                                  3.26.1

                                                                                    Page 9 of 13

Project Research Report: Riko Project  In Riko                     Now
                                       2.6.0 with CUDA 12.6        2.14.1
    Item                               ( install_reqs.sh )
    PyTorch                            Unpinned, base.en on        v1.2.1 (2025-10-31), MIT
    Faster-Whisper                     CPU
    GPT-SoVITS                         API v2 on port 9880         Latest release
                                                                   20250606v2pro; repo still
    VSeeFace                           v1.13.38c β (video)         active (pushed 2026-10-07),
                                                                   MIT
    python-osc                         Version not shown (video)   v1.13.38c5. c4 adds Unity's
    Claude 3.7 Sonnet                  Wrote the avatar            patch for CVE-2025-59489
    OpenRouter                         animations                  (CVSS 8.4)
                                       Used to pick Claude         1.10.2, needs Python 3.10+
     /rankings/programming                                         Retired 2026-02-19
                                       Credited local LLM (video)
    Ana-v1-m7-GGUF                     "Secret link"               Returns 404. /rankings
    Rikoverse                                                      ranks by token usage
                                                                   Still CC-BY-NC-4.0
                                                                   Limited alpha, $7/month
                                                                   early access, "Access opens
                                                                   soon"

7. Best Sources

Primary and official sources only. Issue threads are used for user experience, never for a
verdict.
Checked for this report:

    rayenfeng/riko_project (code, commits, metadata)
    riko_project issues and pull requests
    Rikoverse
    OpenAI model deprecations
    OpenAI gpt-4.1-mini model page

                                                                   Page 10 of 13

Project Research Report: Riko Project

      GPT-SoVITS and its api_v2.py
      Faster-Whisper and on PyPI
      openai on PyPI, torch on PyPI
  Carried over from the video research report:
      VMC protocol specification
      VSeeFace official site and releases
      CVE-2025-59489 (NVD)
      Python 3.10 schedule (PEP 619)
      python-osc on PyPI
      Anthropic: Claude 3.7 Sonnet announcement
      Anthropic model deprecations
      OpenRouter rankings
      TheBloke/Ana-v1-m7-GGUF
      VB-CABLE
      three-vrm

 8. What I Should Actually Believe

  Believe the code you can read and the failures shown on camera. Do not believe the
  README's setup steps as written, or that the public repo is the Riko you see in the videos.
  Definitely true

      The public repo is a working push-to-talk loop: Faster-Whisper, OpenAI gpt-4.1-mini ,
      GPT-SoVITS, with full chat history saved to disk.
       Blend/Val + Blend/Apply and Bone/Pos are real VMC messages, and the video's
      final code uses them correctly.
      ChatGPT's /VMC/Ext/Blend/Expression does not exist.
      The avatar's motion is scripted. Claude 3.7 Sonnet wrote the final idle and talking
      animations.
      The repo has no avatar, VRM, client or local-LLM code, and no commits since 2025-06-
      24.
      Rikoverse is a paid waitlist at $7/month.
  Probably true

                                                                                                                                                                                                                                                  Page 11 of 13

Project Research Report: Riko Project

      The repo runs once you fix the config path, run it from the root, and start GPT-SoVITS
      first.
      The audio-to-avatar pipeline (GPT-SoVITS → VB-CABLE → VSeeFace) works as shown.
      A "private version" exists, given the repo description and the video.
  Questionable
      That the repo is "fully configurable" or uses the GPU for speech recognition.
      That "MIT" applies, with no LICENSE file.
      That the leaderboard shows Claude is "better". It shows usage.
      That "send it twice" is a real fix.
  False
      That config.yaml exists or python main_chat.py works as written.
      That an OSC "Expressions" address is an alternative to VMC.
  Cannot be verified
      Anything about Rikoverse beyond its landing page: features, model, whether it shares
      code with the repo.
      Whether the paid service uses the non-commercial Ana model.
      That the repo was tested on Windows and Ubuntu.
      Who owns the voice in main_sample.wav .

 9. Practical Next Steps

  To run Riko today, fix the config and run command, keep your API key out of git, use a
  supported Python, and build the avatar layer yourself.
  Run the voice loop
   1. Use Python 3.11 or newer (3.10 is end of life) in a fresh virtual environment.
   2. Install and start GPT-SoVITS first: python api_v2.py -a 127.0.0.1 -p 9880 from its

      own folder. An NVIDIA GPU is strongly advised.
   3. In Riko, install extra-req.txt . You only need requirements.txt if GPT-SoVITS shares

      the same environment.
   4. Edit character_config.yaml : set your OpenAI key and change ref_audio_path to

      the full path of character_files/main_sample.wav on your machine.

                                                                                                                                                                                                                                                  Page 12 of 13

Project Research Report: Riko Project

   5. Add a .gitignore with character_config.yaml , chat_history.json and audio/
      before you commit anything. Better still, read the key from an environment variable.

   6. Run python server/main_chat.py from the repo root. Press Enter to start and stop
      recording; keep it under 60 seconds.

   7. For GPU speech recognition, change device="cpu" to "cuda" in main_chat.py .
   8. To avoid OpenAI costs, look at the open Ollama pull request (#16) or point the OpenAI

      client at a local server. Check any local model's license first.
  Add the avatar (not in the repo)
   1. Install VSeeFace v1.13.38c5 or newer (Windows only). Export your VRoid model as

      VRM0.
   2. Use python-osc to send /VMC/Ext/Blend/Val + /VMC/Ext/Blend/Apply for

      expressions and /VMC/Ext/Bone/Pos (all 7 numbers) for bones, to VSeeFace's receiver
      port (default 39539).
   3. Bind any listener to 127.0.0.1 , not 0.0.0.0 , and use different send and receive
      ports.
   4. For lip sync, route GPT-SoVITS audio through VB-CABLE, or send the A/I/U/E/O blend
      shapes directly.
   5. Check every LLM-written OSC address against the VMC spec.
  Before paying for Rikoverse
   1. Ask what it contains and which model it runs. The public repo has no avatar code, and
      the feature claims are unverified.
   2. Compare against the free path: everything above is free except OpenAI tokens and an
      optional VB-CABLE donation.

                                                                                                                                                                                                                                                  Page 13 of 13


```
