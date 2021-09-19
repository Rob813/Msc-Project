
% Create Grid World
GW = createGridWorld(10,10,'Standard') %'kings' - can move in all 8 directions (parameter to change)
GW.CurrentState = '[5,1]';

% Set terminal States and obstacles
GW.TerminalStates = ['[2,9]';'[9,3]'];
GW.ObstacleStates = ["[1,2]";"[1,4]";"[1,6]";"[1,8]";"[2,2]";"[2,4]";"[2,6]";"[2,8]";"[3,2]";"[3,4]";"[3,6]";"[3,8]";"[3,9]";"[5,4]";"[5,5]";"[5,6]";"[5,7]";"[6,4]";"[6,5]";"[6,6]";"[6,7]";"[8,2]";"[8,4]";"[8,6]";"[8,8]";"[9,2]";"[9,4]";"[9,6]";"[9,8]";"[10,2]";"[10,4]";"[10,6]";"[10,8]";]
updateStateTranstionForObstacles(GW)
GW.T(state2idx(GW,"[2,4]"),:,:) = 0;
GW.T(state2idx(GW,"[2,4]"),state2idx(GW,"[4,4]"),:) = 1;

% Define Rewards and Reward Transition Matr, 
nS = numel(GW.States);
nA = numel(GW.Actions);
GW.R = -1*ones(nS,nS,nA);
GW.R(:,state2idx(GW,GW.TerminalStates),:) = 100;
env = rlMDPEnv(GW)

% Start at the same point every time
env.ResetFcn = @() 50;

% Create Agent
qTable = rlTable(getObservationInfo(env),getActionInfo(env));
qRepresentation = rlQValueRepresentation(qTable,getObservationInfo(env),getActionInfo(env));
qRepresentation.Options.LearnRate = 1;

agentOpts = rlQAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = .04;
qAgent = rlQAgent(qRepresentation,agentOpts);

% Train Agent

trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 100;
trainOpts.MaxEpisodes= 200;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 1000;
trainOpts.ScoreAveragingWindowLength = 30;

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end


plot(env)
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;
sim(qAgent,env)