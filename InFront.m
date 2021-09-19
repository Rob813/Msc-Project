%% Grid World with Goal Directly in front of agent

% Create Grid World
GW = createGridWorld(10,10,'kings') %'standard' or 'kings' - can move in either 4 or 8 directions 
%                           ^^ Can also be changed to 'Standard'
GW.CurrentState = '[5,1]';

% Set terminal States and obstacles
GW.TerminalStates = '[1,5]';

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
trainOpts.MaxEpisodes= 20000;
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
