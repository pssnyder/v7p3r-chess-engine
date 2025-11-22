# **v7p3r Lichess Engine: Compute & Deployment Strategy**

This document analyzes the three proposed methods for deploying the v7p3r chess engine on Lichess, evaluates the associated compute costs and performance, and outlines the required play logic and bot configuration.

### **I. Analysis of Deployment Methods**

Each of the three methods offers a distinct balance of cost, control, and complexity.

#### **Method 1: Home Computer Network with Background Nodes**

This method leverages your existing hardware, minimizing upfront costs. It’s a distributed approach where the chess engine's compute tasks are offloaded to background nodes in your personal network.

* **Pros:**  
  * **Minimal Cost:** The primary cost is electricity, as you are utilizing hardware you already own, including a powerful RTX 4070Ti and a high-end i9 CPU.  
  * **High Potential Compute:** The RTX 4070Ti and i9 CPU are significantly more powerful than the RTX 2060 and 4.6GHz CPU proposed in Method 2\. The RTX 4070 Ti is noted to be over 200% faster in synthetic benchmarks than the RTX 2060\.  
  * **Privacy:** All compute is performed on your local network, meaning the data remains within your physical control.  
* **Cons:**  
  * **Complexity:** Architecting a secure, privacy-preserving network of background nodes requires significant technical skill. This includes setting up a robust play server to manage the nodes, securely offloading tasks, and ensuring network privacy for all connected services.  
  * **Unreliability:** The performance and uptime of the bot depend on the availability of your personal computers. If a computer is turned off or put under heavy load from other tasks, the bot's performance could suffer or it could go offline.  
  * **Variable Cost:** While there is no direct "compute" bill, the electricity cost for running powerful CPUs and GPUs around the clock can be substantial and may exceed the monthly budget of a cloud-based service, depending on local energy prices.

#### **Method 2: Dedicated Personal "Server" PC**

This method involves a single, isolated machine dedicated solely to running the chess engine.

* **Pros:**  
  * **Simplicity & Control:** This is the most straightforward and stable on-premise solution. You have complete control over the hardware, software, and environment.  
  * **Security:** Network isolation from your main home network provides a strong security boundary, protecting your private data and other devices.  
  * **Predictable Performance:** The bot’s performance will be consistent since the machine is not burdened by other software or users.  
* **Cons:**  
  * **Upfront Cost:** You would need to purchase or allocate a dedicated PC.  
  * **Single Point of Failure:** If this machine fails or is turned off, the bot is offline. There is no redundancy.  
  * **Suboptimal Compute Utilization:** While the RTX 2060 and 4.6GHz CPU are powerful, they may not be fully utilized at all times, leading to wasted compute resources and consistent electricity costs, regardless of the bot's activity.

#### **Method 3: Purely Cloud-Based Compute**

This approach avoids all hardware and physical infrastructure by running the bot on a cloud provider.

* **Pros:**  
  * **No Hardware Costs:** The most significant advantage is the elimination of upfront hardware investment and ongoing maintenance.  
  * **Scalability & Reliability:** Cloud services are highly reliable and can automatically scale to meet demand, ensuring the bot's performance remains consistent even with a high volume of players.  
  * **Cost-Effective for Low Usage:** You can leverage free tiers and credits from providers like Google Cloud and AWS. For example, Google Cloud offers $300 in free credits for new users, and many services have generous monthly free-tier usage limits. This aligns well with your $20 or less budget, as long as you carefully manage your resource consumption.  
* **Cons:**  
  * **Potential for Cost Overruns:** If not carefully monitored, exceeding free-tier limits can quickly result in a high bill, easily surpassing the $20 monthly budget.  
  * **Less Control:** You have less control over the underlying hardware and software environment compared to an on-premise solution.  
  * **Requires Monitoring:** It is essential to set up cost alerts and usage monitoring to avoid unexpected charges.

### **II. Recommended Strategy & Compute Comparison**

Based on your needs, a hybrid approach could be the most effective.

* **Initial Phase (Development & Testing):** Use **Method 2** (Dedicated Personal Server) with your 4.6GHz CPU/RTX 2060 PC. This provides the ideal environment for development and debugging, offering a secure, stable, and predictable platform to build your engine on. The performance of this machine is more than sufficient for non-simultaneous games and will allow you to get the bot running and tested without the complexity of a distributed system or the risk of unexpected cloud costs.  
* **Expansion Phase (Popularity):** If the bot becomes popular, you can then consider:  
  * **Method 1:** Use your more powerful RTX 4070Ti and i9 machine as a "supernode" to handle longer or more complex matches. This would give you a massive performance boost at no direct cost.  
  * **Method 3:** Move the core "play server" to a cloud provider to handle the queue and game logic, while still using your home machines as worker nodes for the actual engine analysis. This offers the best of both worlds: leveraging your powerful hardware while using a reliable, scalable cloud service for the public-facing aspects.

From a raw compute perspective, your i9/RTX 4070Ti setup is your most powerful asset. The RTX 4070Ti's superior performance in general compute tasks makes it the most capable machine you have for a sophisticated chess engine that will eventually incorporate AI features.

### **III. Lichess Bot Configuration & Play Logic Requirements**

The following requirements must be implemented in the bot's configuration and logic, likely within the lichess-bot framework and a custom script.

#### **1\. Core Play & Queue Logic**

* **Time Controls:** The bot will accept challenges for time controls of 10 minutes or longer with **no increment**. This prevents players from exploiting time controls to win.  
* **Player Queue:** A persistent queue will store players waiting for a match. The queue will intelligently determine when a wait becomes unreasonable (e.g., more than a few hours).  
* **Session Rematch Logic:**  
  * For 10 or 15-minute time controls, the engine will allow up to two rematches from a player (for a total of three games per "session").  
  * After three games, the player must re-add themselves to the queue if other players are waiting.  
  * On slow days with no one in the queue, longer time controls can also allow for rematches.  
* **Traffic-Based Restrictions:**  
  * In a high-traffic scenario (the queue grows to a certain threshold), the bot will limit players to one match per day for time controls longer than 15 minutes.  
  * For 10 or 15-minute time controls, the three-game limit remains, but is also a one-time-per-day restriction.  
* **Skip Logic:** Players will be given a chance to acknowledge their upcoming match. If a player does not acknowledge their match, they will be skipped. After three skips, they will be dropped from the queue.

#### **2\. Admin (v7p3r) Bypass & Testing**

* **Simultaneous Match:** As the admin, you (username v7p3r) must have the ability to initiate a special, non-queued, simultaneous match with the engine.  
* **Zero Disruption:** This match must not disrupt or interfere with any ongoing or queued games for other players. This allows for live debugging and testing of new features.  
* **Debug Mode:** A "breaking bug" must trigger the ability to take the bot offline immediately for repairs without waiting for active games to finish.

#### **3\. Engine Development & Iteration**

The bot is designed to be a progressively updated project.

* **Phase 1: Static Eval:** The initial deployment will be a static evaluation engine.  
* **Phase 2: Heuristics & Enhancements:** Over time, the engine will be progressively updated with new heuristics and features.  
* **Phase 3: AI Features:** The final evolution of the engine will incorporate AI features to provide a varied and dynamic challenge for players.

This document serves as the foundation for your project, providing a clear roadmap for deployment and a checklist of all the necessary game-play logic you've defined.