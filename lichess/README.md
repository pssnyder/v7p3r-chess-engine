<div align="center">

  ![lichess-bot](https://github.com/lichess-bot-devs/lichess-bot-images/blob/main/lichess-bot-icon-400.png)

  <h1>V7P3R Lichess Bot</h1>

  **Production Bot**: [@v7p3r_bot](https://lichess.org/@/v7p3r_bot)  
  **Current Version**: V18.3.0 (PST Optimization)  
  **Status**: ✅ Live on GCP (v7p3r-lichess-bot project)
  
  <br>
  
  **Quick Links:**
  - 📋 [Deployment Quick Reference](DEPLOYMENT_QUICKREF.md) - Fast commands for deployment
  - 📖 [Full Deployment Guide](DEPLOYMENT_GUIDE.md) - Step-by-step with troubleshooting
  - 📝 [CHANGELOG](CHANGELOG.md) - All deployments and version history
  
  <br>
  [![Python Build](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml)
  [![Python Test](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml)

</div>

---

## V7P3R Bot Overview

V7P3R is a UCI-compatible chess engine deployed on [Lichess](https://lichess.org/@/v7p3r_bot) using the [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) framework. The bot runs 24/7 on Google Cloud Platform and competes in rapid, blitz, and bullet games.

### Current Performance
- **ELO Rating**: ~1661 (Rapid, stable)
- **Peak ELO**: 1722 (January 21, 2026)
- **Games Played**: 2,000+ on Lichess
- **Uptime**: 110+ days (v18.3 deployment)

### Key Features
- Adaptive time management with game phase awareness
- Smart search with aspiration windows and early exit
- PST-optimized evaluation (28% speedup)
- Cloud-native deployment on GCP e2-micro instance
- Automated matchmaking and tournament participation

---

## Deployment Documentation

### For Quick Deployment
See **[DEPLOYMENT_QUICKREF.md](DEPLOYMENT_QUICKREF.md)** for copy-paste commands.

### For First-Time Deployment
See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for complete step-by-step guide with:
- Prerequisites and environment setup
- Detailed deployment procedure (9 steps)
- Rollback procedures
- Troubleshooting common issues
- Known pitfalls and solutions

### For Version History
See **[CHANGELOG.md](CHANGELOG.md)** for all deployments, rollbacks, and version notes.

---

## Original lichess-bot Documentation

<div align="center">

  ![lichess-bot](https://github.com/lichess-bot-devs/lichess-bot-images/blob/main/lichess-bot-icon-400.png)

  <h1>lichess-bot</h1>

  A bridge between [lichess.org](https://lichess.org) and bots.
  <br>
  <strong>[Explore lichess-bot docs »](https://github.com/lichess-bot-devs/lichess-bot/wiki)</strong>
  <br>
  <br>
  [![Python Build](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-build.yml)
  [![Python Test](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/python-test.yml)
  [![Mypy](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/mypy.yml/badge.svg)](https://github.com/lichess-bot-devs/lichess-bot/actions/workflows/mypy.yml)

</div>

## Overview

[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) is a free bridge
between the [Lichess Bot API](https://lichess.org/api#tag/Bot) and chess engines.

With lichess-bot, you can create and operate a bot on lichess. Your bot will be able to play against humans and bots alike, and you will be able to view these games live on lichess.

See also the lichess-bot [documentation](https://github.com/lichess-bot-devs/lichess-bot/wiki) for further usage help.

## Features
Supports:
- Every variant and time control
- UCI, XBoard, and Homemade engines
- Matchmaking (challenging other bots)
- Offering Draws and Resigning
- Participating in tournaments
- Accepting move takeback requests from opponents
- Saving games as PGN
- Local & Online Opening Books
- Local & Online Endgame Tablebases

Can run on:
- Python 3.9 and later
- Windows, Linux and MacOS
- Docker

## Steps
1. [Install lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Install)
2. [Create a lichess OAuth token](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token)
3. [Setup the engine](https://github.com/lichess-bot-devs/lichess-bot/wiki/Setup-the-engine)
4. [Configure lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot)
5. [Upgrade to a BOT account](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account)
6. [Run lichess-bot](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Run-lichess%E2%80%90bot)

## Advanced options
- [Create a homemade engine](https://github.com/lichess-bot-devs/lichess-bot/wiki/Create-a-homemade-engine)
- [Add extra customizations](https://github.com/lichess-bot-devs/lichess-bot/wiki/Extra-customizations)

<br />

## Acknowledgements
Thanks to the Lichess team, especially T. Alexander Lystad and Thibault Duplessis for working with the LeelaChessZero team to get this API up. Thanks to [Niklas Fiekas](https://github.com/niklasf) and his [python-chess](https://github.com/niklasf/python-chess) code which allows engine communication seamlessly.

## License
lichess-bot is licensed under the AGPLv3 (or any later version at your option). Check out the [LICENSE file](https://github.com/lichess-bot-devs/lichess-bot/blob/master/LICENSE) for the full text.

## Citation
If this software has been used for research purposes, please cite it using the "Cite this repository" menu on the right sidebar. For more information, check the [CITATION file](https://github.com/lichess-bot-devs/lichess-bot/blob/master/CITATION.cff).
