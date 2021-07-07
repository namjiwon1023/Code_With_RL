# Policy gradients

추천 레퍼런스: 
- https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
- [RLKorea PG여행](https://reinforcement-learning-kr.github.io/2018/06/29/0_pg-travel-guide/)

NPG, TRPO, PPO:
- http://www.andrew.cmu.edu/course/10-703/slides/Lecture_NaturalPolicyGradientsTRPOPPO.pdf
- http://rll.berkeley.edu/deeprlcoursesp17/docs/lec5.pdf

Policy gradient theorem 을 통해 얻을 수 있는 (vanilla) policy gradient 수식은 다음과 같다:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\nabla_\theta J(\pi_\theta)=E_{\tau\sim \pi_\theta} \left[ \sum^T_{t=0} Q^{\pi_\theta} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t|s_t) \right]" style="border:none;">

여기서 여러가지 변주를 줄 수 있는데, GAE 논문에 잘 정리되어 있다.

![pg-gae](https://github.com/namjiwon1023/Code_With_RL/blob/main/assets/rl/pg-gae.png)

여기서 1, 2번이 REINFORCE 에 해당하고, 3번이 REINFORCE with baseline, 4, 5, 6번이 Actor-Critic 에 해당한다.

## REINFORCE

Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

REINFORCE 는 위 피규어에서 1, 2, 3 에 해당하는 방법론이다. 이 세가지 수식이 전부 같은 expectation 값을 갖는데, 앞쪽 수식일수록 variance 가 크다. Q-learning 계열도 마찬가지지만 PG 계열에서도 expectation 을 sampling 으로 대체하게 되는데, 여기서 발생하는 variance 를 잡는 것이 주요한 챌린지가 되며, REINFORCE 에서도 variance 를 줄이기 위한 노력들을 엿볼 수 있다.

<img src="http://chart.googleapis.com/chart?cht=tx&chl=\nabla_\theta J(\pi_\theta)=\mathbb E_{\tau\sim \pi_\theta} \left[ \sum^T_{t=0} (G_t-b(s_t)) \nabla_\theta \log \pi_\theta (a_t|s_t) \right]" style="border:none;">

여기서   <img src="http://chart.googleapis.com/chart?cht=tx&chl=G_t" style="border:none;"> 
는 timestep t 에서의 expected return, <img src="http://chart.googleapis.com/chart?cht=tx&chl=b(s_t)" style="border:none;">  는 baseline 에 해당한다. REINFORCE 는 return G 를 알아야 하기 때문에 하나의 에피소드가 끝나야만 학습을 수행할 수 있다.

## Actor-Critic

Actor-critic 에서는 return G 를 Q-network 으로 approximate 하고, bootstrapping 을 통한 학습을 함으로써 에피소드가 끝나지 않아도 학습이 가능해진다. 이 Q-network 은 actor (policy) 의 행동을 평가하는 역할을 하기 때문에 critic 이라고 부른다.

![pg-ac-alg](https://github.com/namjiwon1023/Code_With_RL/blob/main/assets/rl/pg-ac-alg.png)

Actor-critic 은 이러한 Q actor-critic 외에도 여러 종류가 있다:

![pg-actor-critic](https://github.com/namjiwon1023/Code_With_RL/blob/main/assets/rl/pg-ac.png)
*Image taken from CMU CS10703 lecture slides*

Critic 으로 advantage function A(s,a) 를 사용하는 advantage actor-critic 이 바로 A2C 다. 여기서 여러개의 actor 를 두고 업데이트를 asynchronous 하게 수행하는 A3C 로 발전한다.

### [Trained Results]

![example](./gifs/Pendulum-v0.gif)
