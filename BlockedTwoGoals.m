% Grid World with two blocked goals to the right and left of agent

% Create Grid World
GW = createGridWorld(10,10,'Kings') %'kings' - can move in all 8 directions (parameter to change)
%                           ^^ Can also be changed to 'Standard'
% Set terminal States and obstacles
GW.TerminalStates = ['[1,2]';'[1,9]'];
GW.ObstacleStates = ['[1,3]';'[2,3]';'[3,2]';'[1,8]';'[2,8]';'[3,9]'];
updateStateTranstionForObstacles(GW)

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
trainOpts.StopTrainingValue = 1820;
trainOpts.ScoreAveragingWindowLength = 20;

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
