#major_imports
from glob import glob 
from klampt import IKObjective
from klampt import WorldModel,Geometry3D
from klampt import vis
from klampt.math import so3,se3
from klampt.io import resource
from klampt.math import vectorops,so3
from klampt.model.collide import WorldCollider
from klampt.model import ik
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.model.trajectory import RobotTrajectory,Trajectory
from klampt.math.vectorops import interpolate
from klampt.plan.cspace import CSpace,MotionPlan
from planning.disinfection3d import DisinfectionProblem
from planning.robot_cspaces import Robot3DCSpace,CSpaceObstacleSolver,UnreachablePointsError
import pickle
import os
import numpy as np
from planning.tsp_solver_wrapper import runTSP
import networkx as nx
import klampt
from klampt.plan.robotcspace import RobotCSpace
from klampt.plan import cspace
from klampt.model import collide
import klampt
from klampt.plan import cspace,robotplanning
from klampt.io import resource
import time
import trimesh as tm
from alphashape import alphashape
import shapely
from planning.auxiliary_functions import setup_robot_and_light,get_bounds,load_relevant_info
from planning.auxiliary_functions import extract_milestone_tour,collisionChecker,find_n_ik_solutions
from planning.auxiliary_functions import iterative_farthest_point,euclidean_trelis_solution,set_robot_link_collision_margins
from planning.auxiliary_functions import interp,compute_actual_distance,heuristic_penalized_trelis_solution
from planning.auxiliary_functions import heuristic_ee_trelis_solution
from tqdm import tqdm
from random import choice
from klampt.model.trajectory import RobotTrajectory,Trajectory
from planning.robot_cspaces import Robot3DCSpace,UnreachablePointsError,CSpaceObstacleSolver

roadmaps = glob('./Distance/*/*/*/armbot_roadmap_330_divs.p')

robot_name = 'armbot'
tmin = 0.5
world,robot,lamp,roadmap,sampling_places,adjacency_matrix,node_coords,solutions,bounds,reachable,configs,alpha_shape,collider = load_relevant_info(roadmaps[0],robot_name = 'armbot')
selected_points = sampling_places[reachable,:][solutions>tmin,:]
total_dofs  = len(robot.getConfig())

points_mask = solutions>tmin
# final_trajectory,final_cost = extract_milestone_tour(adjacency_matrix,robot_name,sampling_places,configs,reachable,solutions,node_coords,roadmap = roadmap,full_trajectory = False)
rrt_space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.1,
                                    ignoreCollisions = [(robot.link(2),robot.link(3)),
                                                    (robot.link(8),robot.link(6))])
load = True

if not load:
    ik_sols = []
    for point in tqdm(selected_points):
        candidate_ik_sols = find_n_ik_solutions(robot,lamp,collider,world,point,alpha_shape,
                                      restarts = 100,tol = 1e-3,neighborhood = 0.4,float_height = 0.15, n = 2000)
        subsample = iterative_farthest_point(candidate_ik_sols,200)
        ik_sols.append(subsample)
    pickle.dump(ik_sols,open('ik_sols_200.p','wb'))
else:
    ik_sols = pickle.load(open('ik_sols_200.p','rb'))
set_robot_link_collision_margins(robot,0,collider)
base = robot.link(2)
base.geometry().setCollisionMargin(0)
sols = 200

milestones = []
for ik_sol in ik_sols:
    this_conf = ik_sol[0]
    robot.setConfig(this_conf)
    pos = np.array(lamp.getWorldPosition([0,0,0]))
    new_ik_sol = np.zeros((ik_sol.shape[0],ik_sol.shape[1]+3))
    new_ik_sol[:,:ik_sol.shape[1]] = ik_sol
    new_ik_sol[:,ik_sol.shape[1]:] = pos
    milestones.append(new_ik_sol)
milestones = np.array(milestones)
milestones = milestones.reshape(-1,ik_sols[0].shape[-1]+3)
space = Robot3DCSpace(bounds,robot,collider,lamp,milestones,
    base_height_link = 2,
    robot_height = 1.5,
    float_height = 0.15,
    linear_dofs = [0,1],
    angular_dofs = [4,5,6,7,8,9],
    light_local_position  = [0,0,0])
program = CSpaceObstacleSolver(space,milestones = milestones, initial_points= 4000,steps = 100,max_iters = 15000)
adjacency_matrix,roadmap,node_coords = program.get_adjacency_matrix_from_milestones()

full_distances = adjacency_matrix

pickle.dump(full_distances,open('200x200x86_distances.p','wb'))

# penalty = 2
# coarse_space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.5,
#                                     ignoreCollisions = [(robot.link(2),robot.link(3)),
#                                                         (robot.link(8),robot.link(6))])
# distances = np.zeros((adjacency_matrix.shape[0]+1,adjacency_matrix.shape[0]+1))
# distances[1:,1:] = 1000*adjacency_matrix
# euc_distances = distances
# tour = runTSP(euc_distances, '/{}_currTSP'.format(robot_name)) 
# tour = (np.array(tour[1:])-1).tolist()
# trelis_ik = np.array(ik_sols)[tour]
# best_indices = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
# best_cost = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
# for i in tqdm(range(trelis_ik.shape[0]-1)):
#     current_iks = trelis_ik[i]
#     next_iks = trelis_ik[i+1]
#     actual_i = tour.index(i)
#     actual_next_i = tour.index(i+1)
#     approx_dist = full_distances[sols*(actual_i):sols*(actual_i+1),sols*(actual_next_i):sols*(actual_next_i+1)]
# #     break
# #     for j,point in enumerate(next_iks):
# #         for k,cur_point in enumerate(current_iks):
# # #             interpolated_cfigs = interp(cur_point,end,robot)
# #             if(coarse_space.isVisible(cur_point,point)):
# #                 approx_dist[j,k] = compute_actual_distance(cur_point,point,lamp,robot)
# #             else:
# #                 approx_dist[j,k] = penalty*compute_actual_distance(cur_point,point,lamp,robot)
# #             approx_dist[j,:] = np.linalg.norm(current_iks-point,axis = 1)

#     best_idx = np.argmin(approx_dist,axis = 1)
#     cost = np.min(approx_dist, axis = 1)
#     if(i != 0):
# #         print(best_cost[i-1,best_idx])
#         cost += best_cost[i-1,best_idx]
#     best_indices[i,:] = best_idx
#     best_cost[i,:] = cost
# j = best_indices.shape[0]-1
# best_idx_path = []
# best_idx_path.append(np.argmin(best_cost[j]).astype(int))
# while(j>=0):
#     best_idx_path.append(best_indices[j,best_idx_path[-1]].astype(int))
#     j -= 1
# best_idx_path.reverse()
# configuration_sequence = []
# for i,idx in enumerate(best_idx_path):
#     configuration_sequence.append(trelis_ik[i,idx])

# rrt_space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.1,
#                                     ignoreCollisions = [(robot.link(2),robot.link(3)),
#                                                     (robot.link(8),robot.link(6))])
# final_path = []
# for i in tqdm(range(len(configuration_sequence)-1)):
#     curr_cfig = configuration_sequence[i]
#     next_cfig = configuration_sequence[i+1]
#     if(rrt_space.isVisible(curr_cfig,next_cfig)):
#         path = [curr_cfig,next_cfig]
#     else:
#         planner = cspace.MotionPlan(rrt_space,type="rrt*",connectionThreshold=50.0,bidirectional = 1,shortcut = True,knn = 60)  #accepts keyword arguments
#         planner.setEndpoints(curr_cfig,next_cfig)
#         increment = 500               #there is a little overhead for each planMore call, so it is best not to set the increment too low
#         t0 = time.time()
#         tmax = 2
#         no_path = True
#         while time.time() - t0 < tmax or no_path:   #max 20 seconds of planning
#             planner.planMore(increment)
#             path = planner.getPath()
#             if(path is None):
#                 if(time.time() - t0 > 10*tmax):
#     #                 print('hmmmm planning failed, extending time by 5 seconds')
#                     next_cfig = choice(ik_sols[i+1])
#                     configuration_sequence[i+1] = next_cfig
#                     planner.close()
#                     planner = cspace.MotionPlan(rrt_space,type="rrt*",connectionThreshold=50.0,bidirectional = 1,shortcut = False,knn = 60)  #accepts keyword arguments
#                     planner.setEndpoints(curr_cfig,next_cfig)
#                     print(next_cfig)
#             else:
#                 no_path = False
#     if(i != len(configuration_sequence)-1):
# #         print(path)
#         final_path.append(path[:-1])
#     else:
#         final_path.append(path)
#     #     if path is not None:
#     #         print("Solved, path has",len(path),"milestones")
#     #         print("Took time",time.time()-t0)
#     #         break

#     planner.close() 



# appended_path = []
# for i in final_path:
#     appended_path.extend(i)
# total_distance = 0 
# planner = cspace.MotionPlan(rrt_space,type="rrt*",connectionThreshold=50.0,bidirectional = 1,shortcut = False,knn = 30)  #accepts keyword arguments

# for i in range(len(appended_path)-1):
#     origin = appended_path[i]
#     end = appended_path[i+1]
#     total_distance += compute_actual_distance(origin,end,lamp,robot)
# total_distance
