import numpy as np
import enum
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from fitQ import fitQ, loglikelihood
from sklearn.feature_extraction import DictVectorizer
import statsmodels.api as sm
from collections import defaultdict
import pandas as pd

# delay (door closed, bridge closed, no living penalty) 
# -delay time up-> bridge open, door still closed 
# --> approach and choice 
# --> reach goal, door open, bridge close
# --> return to origin
# --> bridge close

# 1 1 1 1 1 1 1 
# 1 0 0 B 0 0 1
# 1 0 1 0 1 0 1
# 1 0 1 x 1 0 1
# 1 0 0 D 0 0 1
# 1 1 1 1 1 1 1 

class Stage(enum.IntEnum):
    APPROACH = 0;
    RETURN = 1;

class TMaze:
    def __init__(self, height, width, max_steps, switch_after, obs_window_size=1):
        self.height = height;
        self.width = width;
        self.maze = np.ones((height+2, width*2+3));
        self.obs_window_size = 1; # the size of window of observation is (2*obs_window_size+1)^2;
        self.max_steps = max_steps; # maximum number of steps each episode
        self.switch_after = switch_after; # switch goal after this many number of goals

        # build maze
        self.maze[1, 1:-1] = 0;
        self.maze[-2, 1:-1] = 0;
        self.maze[1:-1, 1] = 0;
        self.maze[1:-1, width+1] = 0;
        self.maze[1:-1, -2] = 0;

        self.origin = np.array([self.height-1, self.width+1]);
        self.choice_point = np.array([1, self.width+1]);
        self.left_goal = np.array([1, self.width]);
        self.right_goal = np.array([1, self.width+2]);

        self.legal_positions = {};
        for i in range(1, self.height+1):
            for j in range(1, 2*self.width+3):
                if (self.maze[i,j]==0):
                    self.legal_positions[(i, j)] = 0;
        self.num_legal_positions = self.height*3+4*self.width-4;
        assert(len(self.legal_positions)==self.num_legal_positions);

        self.mirror_positions = {};
        for x in self.legal_positions.keys():
            if x[1]>self.width and (self.maze[x[0], 2*self.width+2-x[1]]==0):
                self.mirror_positions[x] = (x[0], 2*self.width+2-x[1]);
            else:
                self.mirror_positions[x] = x;

        self.agentpos = self.origin.copy(); # initialize at bottom of maze
        self.t = 0; # time within each trial
        self.epNum = 0; # the number of times the goal is reached
        self.stage = Stage.APPROACH;

        self.action2movment = [
            np.array([0, -1]), #left
            np.array([0, +1]), #right
            np.array([+1, 0]), #down
            np.array([-1, 0]), #up
        ];

        self.all_goal_probs = [
            [0.85, 0.15],
            [0.15, 0.85],
        ];
        
        self._sample_task();
        self._open_bridge(); 
        self._close_door();

        # for plotting
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

    def step(self, action):
        # calculate new position
        newPos = self.agentpos+self.action2movment[action];
        done = False;
        reward = 0.0;
        episode_done = False;
        which_goal = None;
        rewarded = False;
        self.t += 1;

        if (self.t==self.max_steps):
            done = True;

        if (self.stage==Stage.APPROACH):
            if self.maze[newPos[0], newPos[1]]==1:
                reward = -0.01; # if not legal move, incur penalty; else no penalty
            else:
                self.agentpos = newPos; # else make the move
                if (np.array_equal(self.agentpos, self.left_goal) or np.array_equal(self.agentpos, self.right_goal)):
                    which_goal = "left" if np.array_equal(self.agentpos, self.left_goal) else "right";
                    if ((np.array_equal(self.agentpos, self.left_goal) and np.random.rand(1)<self.goal_probs[0]) or 
                        (np.array_equal(self.agentpos, self.right_goal) and np.random.rand(1)<self.goal_probs[1])):
                        reward = +1.0; # give reward if arrive at food goal
                        rewarded = True;
                    self.epNum += 1; # count the number of times a goal is reached
                    episode_done = True;
                    self._open_door(); # open the door back
                    self._close_bridge(); # close the bridge so that agent doesn't circle back
                    self.stage = Stage.RETURN;
                else:
                    reward = -0.005; # small living penalty to encourage short path
        elif (self.stage==Stage.RETURN):
            if self.maze[newPos[0], newPos[1]]==1:
                reward = -0.01; # if not legal move, incur penalty; else no penalty
            else:
                self.agentpos = newPos; # else make the move
                if (np.array_equal(self.agentpos, self.origin)):
                    reward = +1.0; # give reward if arrive at origin
                    if ((self.epNum+1)%self.switch_after==0):
                        self._sample_task();
                    self._close_door();
                    self._open_bridge();
                    self.stage = Stage.APPROACH;                    
                else:
                    reward = -0.005;
            # {"time": self.t, "stage": self.stage, "pos": self.agentpos, "goal_probs": self.goal_probs, "episode_done": episode_done};

        return self._get_obs(), reward, done, \
            {"pos": tuple(self.agentpos.copy()), \
             "choice_point": np.array_equal(self.agentpos, self.choice_point), \
             "origin": np.array_equal(self.agentpos, self.origin), \
             "stage": self.stage,\
             "which_goal": which_goal,\
             "rewarded": rewarded,\
             "which_task": self.all_goal_probs.index(self.goal_probs)}

    def reset(self):
        self.agentpos = self.origin.copy(); # initialize at bottom of maze
        self.t = 0; # time within each trial
        self.epNum = 0; # the number of times the goal is reached
        self.stage = Stage.APPROACH;
        self._sample_task();
        self._open_bridge(); 
        self._close_door();

    def _get_obs(self):
        return self.maze[
                    self.agentpos[0]-self.obs_window_size : self.agentpos[0]+self.obs_window_size+1, 
                    self.agentpos[1]-self.obs_window_size : self.agentpos[1]+self.obs_window_size+1
                ];
        
    def _open_bridge(self):
        self.maze[1, self.width+1] = 0;
    
    def _close_bridge(self):
        self.maze[1, self.width+1] = 1;

    def _open_door(self):
        self.maze[-2, self.width+1] = 0;
    
    def _close_door(self):
        self.maze[-2, self.width+1] = 1;

    def _sample_task(self):
        self.goal_probs = self.all_goal_probs[np.random.randint(len(self.all_goal_probs))];

    def encode_pos(self, pos):
        enc = self.legal_positions.copy();
        enc[pos] = 1.0
        return enc;

    def render(self, close=False):
        if close:
            plt.close()
            return
                
        obs = self.maze.copy();
        partial_obs = self._get_obs().copy();

        obs[self.agentpos[0], self.agentpos[1]] = 2.0;
        partial_obs[self.obs_window_size, self.obs_window_size] = 2.0;

        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
        self.ax_full.axis('off')
        self.ax_partial.axis('off')
        
        self.fig.show()

        # Only create the image the first time
        if not hasattr(self, 'ax_full_img'):
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
        if not hasattr(self, 'ax_partial_img'):
            self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
        # Update the image data for efficient live video
        self.ax_full_img.set_data(obs)
        self.ax_partial_img.set_data(partial_obs)
        
        plt.draw();
        self.fig.canvas.draw()

        return self.fig

# a trajectory should record
# 1. network activities (nope)
# 2. position
# 3. whether it is at choice point
# 4. whether it is at left/right goal
# 5. which stage
# 6. did it get food

# we do this for multiple trajectories: traj should be a 2d array

class Trajectories:
    def __init__(self, maze):
        self.traj = [];
        self.maze = maze;
    
    def add_info(self, info):
        assert(set(("pos", "choice_point", "origin", "which_goal", "stage", "rewarded", "which_task"))==info.keys());
        self.traj[-1].append(info);
    
    def add_new_episode(self):
        self.traj.append([{
             "pos": tuple(self.maze.origin.copy()), \
             "choice_point": False, \
             "origin": True, \
             "stage": Stage.APPROACH,\
             "which_goal": None,\
             "rewarded": False,\
             "which_task": self.maze.all_goal_probs.index(self.maze.goal_probs),\
        }]);
    
    def reset(self):
        self.traj[:] = [];

    def get_choices_and_outcome(self):
        choices = [];
        outcomes = [];
        for t in self.traj:
            choices.append([]);
            outcomes.append([]);
            for step in t:
                if (step["which_goal"]!=None):
                    outcomes[-1].append(1.0 if step["rewarded"] else 0.0);
                    choices[-1].append(step["which_goal"]);

        return choices, outcomes;

    def get_choice_times(self):
        times = [];
        for t in self.traj:
            times.append([]);
            for i in range(len(t)):
                if (t[i]["choice_point"]):
                    times.append(i);
        return times;

    def get_goal_times(self):
        times = [];
        for t in self.traj:
            times.append([]);
            for i in range(len(t)):
                if (t[i]["which_goal"]!=None):
                    times[-1].append(i);
        return times;

    def get_origin_times(self):
        times = [];
        for t in self.traj:
            times.append([]);
            prev_trial_ended = True;
            for i in range(len(t)):
                if (t[i]["origin"]) and prev_trial_ended:
                    times[-1].append(i);
                    prev_trial_ended = False;
                elif (t[i]["which_goal"]!=None):
                    prev_trial_ended = True
        return times;

    def segment_states_by_trial_time(self, states):
        # states should be of size: num trajectories
        seged_states = []; 
        origin_times = self.get_origin_times(); # get the timesteps where the agent is at the origin
        for i in range(len(states)):
            # for each trajectory
            seged_traj = [];
            # append each subarray between two timestpes where the agent is at the origin
            for j in range(len(origin_times)-1):
                seged_traj.append(states[i][origin_times[j]:origin_times[j+1]]);
            seged_states.append(seged_traj);
        # seged_states should be of size: num trajectories, num trials in each trajectory, num neurons
        return seged_states;

    def segment_pos_by_trial_time(self, pos):
        seged_poses = []; 
        origin_times = self.get_origin_times(); # get the timesteps where the agent is at the origin
        for i in range(len(pos)):
            # for each trajectory
            seged_traj = [];
            # append each subarray between two timestpes where the agent is at the origin
            for j in range(len(origin_times)-1):
                seged_traj.append(pos[i][origin_times[j]:origin_times[j+1]]);
            seged_poses.append(seged_traj);
        # seged_states should be of size: num trajectories, num trials in each trajectory, num neurons
        return seged_poses;

    def get_pos_one_hot(self):
        pos_encoded = [];
        for t in self.traj:
            pos_encoded.append([]);
            for i in range(len(t)):
                pos_encoded[-1].append(self.maze.encode_pos(t[i]["pos"]));
        return pos_encoded;

    def get_task(self):
        tasks = [];
        for t in self.traj:
            tasks.append([]);
            for i in range(len(t)):
                tasks[-1].append(t[i]["which_task"]);
        return tasks;

    def get_stage(self):
        stgs = [];
        for t in self.traj:
            stgs.append([]);
            for i in range(len(t)):
                stgs[-1].append(int(t[i]["stage"]));
        return stgs;

    def get_feats(self, Qls, Qrs, abe, acts):
        '''
            activity ~ R(t-1) + C(t-1) + C(t) + dQ(t) + Qc(t-1)
        '''        
        feats = defaultdict(list);
        pos = [[s["pos"] for s in t] for t in self.traj];
        choices, outcomes = self.get_choices_and_outcome();
        goal_times = self.get_goal_times();
        print([len(t) for t in goal_times])
        tasks = self.get_task()
        for t in range(len(self.traj)):
            #for each trajectory, for each timestep in that trajectory, there should be a feature vector corresponding to that 
            for i in range(len(goal_times[t])-1):
                for j in range(goal_times[t][i], goal_times[t][i+1]):
                    feats[self.maze.mirror_positions[pos[t][j]]].append({
                        "art-1": acts[t][j-1],
                        "art-2": acts[t][j-2],
                        "art-3": acts[t][j-3],
                        "prev_choice": 1.0 if choices[t][i]=="left" else 0.0,
                        "task_type": tasks[t][i],
                        "rpe": outcomes[t][i]-(Qls[t][i] if choices[t][i]=="left" else Qrs[t][i]),
                        "updated_qc": abe[0]*outcomes[t][i]+(1-abe[0])*(Qls[t][i] if choices[t][i]=="left" else Qrs[t][i]),
                        "upcoming_choice": 1.0 if choices[t][i+1]=="left" else 0.0,
                        "qdiff": Qls[t][i+1]-Qrs[t][i+1],
                        "qsum": Qls[t][i+1]+Qrs[t][i+1],
                        "q_upcoming_choice": Qls[t][i+1] if choices[t][i+1]=="left" else Qrs[t][i+1],
                    });
                        # "qsum": Qls[t][i+1]+Qrs[t][i+1],
                        # "q_prev_c": Qls[t][i] if choices[t][i]=="left" else Qrs[t][i],
                        # "prev_outcome": outcomes[t][i],
                        # "choice_X_outcome": outcomes[t][i] * (1.0 if choices[t][i]=="left" else 0.0),

        return feats;

    def linear_regression_fit(self, feats, activities):
        # get the right activities
        goal_times = self.get_goal_times();
        pos = [[s["pos"] for s in t] for t in self.traj];
        activities_by_pos = defaultdict(list);

        for t in range(len(activities)):
            for j in range(goal_times[t][0], goal_times[t][-1]):
                activities_by_pos[self.maze.mirror_positions[pos[t][j]]].append(activities[t][j]);

        # flatten all 
        acts_flat = {};
        for k, v in activities_by_pos.items():
            acts_flat[k] = np.array(v);
            acts_flat[k] = (acts_flat[k]-acts_flat[k].mean(axis=0))/(acts_flat[k].std(axis=0)+1e-6)

        all_feats = [entry for f in feats.copy().values() for entry in f];

        dictvec = DictVectorizer(sort=False, sparse=False);
        feat_fit = dictvec.fit(all_feats);
        headers = ['Intercept']
        headers.extend(dictvec.feature_names_);

        feats_flat = {};
        for k, v in feats.items():
            feats_flat[k] = feat_fit.transform(v);
            feats_flat[k] = (feats_flat[k]-feats_flat[k].mean(axis=0))/(feats_flat[k].std(axis=0)+1e-6)
            feats_flat[k] = sm.add_constant(feats_flat[k], prepend=True, has_constant="raise");
            feats_flat[k] = pd.DataFrame(feats_flat[k], columns=headers)

        results = defaultdict(list);

        for k in feats_flat.keys():
            for n in range((len(activities[0][0]))):
                results[k].append(sm.OLS(acts_flat[k][:,n], feats_flat[k]).fit());

        return results, dictvec, feats_flat, acts_flat;