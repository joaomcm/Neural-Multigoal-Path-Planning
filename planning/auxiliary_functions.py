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
from random import choice
from tqdm import tqdm
def setup_robot_and_light(robotfile = './data/armbot.rob',
                                mesh_file = './full_detail_hospital_cad_meters.obj',float_height = 0.08,bounds = []):
    world = WorldModel()
    res1 = world.loadElement(robotfile)
    robot = world.robot(0)
    limits = robot.getJointLimits()
    limits[0][0] = bounds[0]
    limits[0][1] = bounds[1]
    limits[1][0] = bounds[2]
    limits[1][1] = bounds[3]
    limits[0][2] = float_height
    limits[1][2] = float_height
    robot.setJointLimits(limits[0],limits[1])
    #world.loadElement(robotfile)

    #a = Geometry3D()
    # res = a.loadFile(mesh_file)
    res = world.loadElement(mesh_file)
    print(res)
    collider = WorldCollider(world)
    #we then dilate the base nd ignore collisions between it and the 2 subsequent links:
    collider.ignoreCollision((robot.link(2),robot.link(3)))
    collider.ignoreCollision((robot.link(2),robot.link(3)))
    collider.ignoreCollision((robot.link(8),robot.link(6)))
    # we now 
    cfig = robot.getConfig()
    terrain = world.terrain(0)
    set_robot_link_collision_margins(robot,0.03,collider)
    base = robot.link(2)
    base.geometry().setCollisionMargin(0.03)
    lamp = robot.link(11)
    print('\n\n\nbase height link = {}, lamp linknum = {}\n\n\n'.format(2,1))
    cfig[2] = float_height
    robot.setConfig(cfig)
    robot.link(11).appearance().setColor(210/255,128/255,240/255,1)

    world.saveFile('disinfection.xml')
    return world,robot,lamp,collider



def get_maximum_connected_component_mesh(mesh_file):
    full_mesh = tm.exchange.load.load(mesh_file)
    full_mesh.show()
    conn_components = tm.graph.split(full_mesh, only_watertight = False)
    max_faces = 0
    for component in conn_components:
        if(component.faces.shape[0]>max_faces):
            max_faces = component.faces.shape[0]
            maximum_connected_component  = component
    return maximum_connected_component


def get_bounds(mesh_file,degree = 10):
    mesh = get_maximum_connected_component_mesh(mesh_file)
    projection = mesh.vertices[mesh.vertices[:,2]<0.3,:2]
    projection = np.unique(projection,axis = 0).astype(float)
    alpha_shape = alphashape(projection,degree)
    # if the alpha_shape is a multipoligon, we get its component with largest area
    if(type(alpha_shape) == shapely.geometry.multipolygon.MultiPolygon):
        max_area = 0
        for shape in alpha_shape:
            if(shape.area>max_area):
                max_area = shape.area
                final_shape = shape
        alpha_shape = final_shape
    bounds = alpha_shape.bounds
    
    return bounds,alpha_shape


def load_relevant_info(rm_file,robot_name = 'armbot'):
    robot_name = 'armbot'
    total_dofs = 11
    main_dir = os.path.dirname(rm_file)
    mesh_name = os.path.basename(os.path.dirname(os.path.dirname(main_dir)))
    mesh_file = './data/icra_mesh_final/{}.ply'.format(mesh_name)
    bounds,alpha_shape = get_bounds(mesh_file)
    world,robot,lamp,collider = setup_robot_and_light(mesh_file = mesh_file,float_height = 0.15,bounds = bounds)
    reachable_file = os.path.join(main_dir,'{}_reachable_330_divs.p'.format(robot_name))
    solutions_file = os.path.join(main_dir,'{}_solutions_330_divs.p'.format(robot_name))
    configs_file = os.path.join(main_dir,'{}_configs_330_divs.p'.format(robot_name))
    roadmap = pickle.load(open(rm_file,'rb'))
    configs = pickle.load(open(configs_file,'rb'))
    sampling_places_file = os.path.join(main_dir,'{}_sampling_places_330_divs.p'.format(robot_name))
    adjacency_file =  os.path.join(main_dir,'{}_adjacency_330_divs.p'.format(robot_name))
    node_coords_file = os.path.join(main_dir,'{}_node_coords_330_divs.p'.format(robot_name))
    reachable = pickle.load(open(reachable_file,'rb'))
    solutions = np.array(pickle.load(open(solutions_file,'rb'))[0])
    sampling_places = pickle.load(open(sampling_places_file,'rb'))
    adjacency_matrix = pickle.load(open(adjacency_file,'rb'))
    node_coords = pickle.load(open(node_coords_file,'rb'))
    return world,robot,lamp,roadmap,sampling_places,adjacency_matrix,node_coords,solutions,bounds,reachable,configs,alpha_shape,collider


def extract_milestone_tour(adjacency_matrix,robot_name,sampling_places,configs,reachable,solutions,node_coords,roadmap = None,full_trajectory = False):
    # we then solve the TSP:
    selected_points = sampling_places[reachable,:][solutions>0,:]
    points_mask = solutions>0.5
    
    distances = np.zeros(shape = (adjacency_matrix.shape[0]+1,adjacency_matrix.shape[0]+1))
    distances[1:,1:] = 100*adjacency_matrix.copy()
    
    tour = runTSP(distances, '/{}_currTSP'.format(robot_name)) # We just have an arbitrary name since it doesn't matter - can change this so that user can input filename if desired
    tour = (np.array(tour[1:])-1).tolist()
    indices = np.array(range(sampling_places[reachable,:].shape[0]))
    used_indices = indices[points_mask]
    tour_indices = used_indices[tour]

    # We then calculate the total distance travelled - and compute final trajectory:
    total_distance = 0 
    final_trajectory = []
    if(full_trajectory):
        for i in range(len(tour)-1):
            current_point = tour[i]
            next_point = tour[i+1]
            total_distance += adjacency_matrix[current_point,next_point]
            nodes_path = nx.algorithms.shortest_path(roadmap,source = current_point,target = next_point,weight = 'weight')
        #     print(nodes_path)
        #     print(nodes_path)
            traj = [node_coords[j][:11+1] for j in nodes_path]
        #     print(trajectory)
            if(i != len(tour)-1):
                final_trajectory.append(traj[:-1])
            else:
                final_trajectory.append(traj)
        return final_trajectory,total_distance
    else:
        filtered_configs = np.array(configs)[reachable,:][solutions>0,:]
        final_trajectory = []
        for i in range(len(tour)):
            final_trajectory.append(filtered_configs[i])
        return final_trajectory,total_distance

def set_robot_link_collision_margins(robot,margin,collider,range_adjust_collision = [4,5,6,7,8]):
    for link_num in range_adjust_collision:
        this_link = robot.link(link_num)
        this_link.geometry().setCollisionMargin(margin)
        collider.ignoreCollision((robot.link(link_num),robot.link(link_num+1)))


def collisionChecker(collider,robot,alpha_shape,base_radius = 0.225):
        base_radius = 0.225
        base_center = shapely.geometry.Point(robot.getConfig()[:2])
        within_boundary = alpha_shape.contains(base_center)
        # print(within_boundary)
        if(within_boundary == False):
            # print('\n\n\n out of bounds')
            # point is not interior to the alpha shape of the floorplan
            return False
        elif(alpha_shape.exterior.distance(base_center) < base_radius):
            # point is interior to the alphashape, but base is partially outside of it.
            return False
        elif(list(collider.collisions())!= []):
            # print('\n\n\n New Collision \n\n')
            # for i in collider.collisions():
            #     print(i[0].getName(),i[1].getName())
            # print(list(collider.collisions()))
            return False
        return True

def find_n_ik_solutions(robot,lamp,collider,world,place,alpha_shape,restarts = 100,tol = 1e-3,neighborhood = 0.4,float_height = 0.15,active_dofs = [0,1,4,5,6,7,8,9],n = 20):
    solutions =[]
    while(len(solutions) < n):
        goal = place.tolist()
        obj = ik.objective(lamp,local = [0,0,0], world = goal)
        solver = ik.solver(obj)
        solver.setMaxIters(100)
        solver.setTolerance(tol)
        jl = solver.getJointLimits()
        jl[0][0] = goal[0] - neighborhood
        jl[1][0] = goal[0] + neighborhood
        jl[0][2] = float_height
        jl[0][1] = goal[1] - neighborhood
        jl[1][1] = goal[1] + neighborhood
        jl[1][2] = float_height
        solver.setJointLimits(jl[0],jl[1])
        solver.setActiveDofs(active_dofs)
        solver.sampleInitial()

        for i in range(restarts):
            if(solver.solve()):
                if(collisionChecker(collider,robot,alpha_shape)):
                    solutions.append(robot.getConfig())
                    break
                    
                else:
                    solver.sampleInitial()
            else:
                solver.sampleInitial()
    return solutions

def iterative_farthest_point(ik_sols, n = 20):
    first_point = np.array(choice(ik_sols))
    ik_sols = np.array(ik_sols)
    selected_points = [first_point]
    distances = np.zeros(ik_sols.shape[0])
    distances[:] = np.linalg.norm(ik_sols-first_point,axis = 1)
    while (len(selected_points) < n):
        next_point = ik_sols[np.argmax(distances),:]
        selected_points.append(next_point)
        new_distances = np.linalg.norm(ik_sols-next_point,axis = 1)
        closer = new_distances < distances
        distances[closer] = new_distances[closer]
    return np.array(selected_points)


def euclidean_trelis_solution(selected_points,ik_sols,robot_name):
    distances = np.zeros((selected_points.shape[0]+1,selected_points.shape[0]+1))
    for i in range(selected_points.shape[0]):
        point = selected_points[i]
        distances[1:,i+1] = 1000*np.linalg.norm(selected_points-point,axis = 1)
    euc_distances = distances
    tour = runTSP(euc_distances, '/{}_currTSP'.format(robot_name)) 
    tour = (np.array(tour[1:])-1).tolist()
    trelis_ik = np.array(ik_sols)[tour]
    best_indices = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    best_cost = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    for i in range(trelis_ik.shape[0]-1):
        current_iks = trelis_ik[i]
        next_iks = trelis_ik[i+1]
        approx_dist = np.zeros((trelis_ik.shape[1],trelis_ik.shape[1]))
        for j,point in enumerate(next_iks):
            approx_dist[j,:] = np.linalg.norm(current_iks-point,axis = 1)

        best_idx = np.argmin(approx_dist,axis = 1)
        cost = np.min(approx_dist, axis = 1)
        if(i != 0):
    #         print(best_cost[i-1,best_idx])
            cost += best_cost[i-1,best_idx]
        best_indices[i,:] = best_idx
        best_cost[i,:] = cost
    j = best_indices.shape[0]-1
    best_idx_path = []
    best_idx_path.append(np.argmin(best_cost[j]).astype(int))
    while(j>=0):
        best_idx_path.append(best_indices[j,best_idx_path[-1]].astype(int))
        j -= 1
    best_idx_path.reverse()
    configuration_sequence = []
    for i,idx in enumerate(best_idx_path):
        configuration_sequence.append(trelis_ik[i,idx])
    return configuration_sequence,trelis_ik

def compute_actual_distance(origin,end,lamp,robot):
    #         for origin,end in zip(origins,ends):
    interpolated_cfigs = interp(origin,end,robot)
    positions = []
    for cfig in interpolated_cfigs:
        robot.setConfig(cfig)
        positions.append(lamp.getWorldPosition([0,0,0]))
    positions = np.array(positions)
    distance = np.linalg.norm(np.diff(positions,axis = 0),axis = 1).sum()
    return distance

def interp(m_a,m_b,robot,steps = 25):
    divs = np.linspace(0,1,num = steps)
    # dif = m_b-m_a
    # interm_steps = m_a + np.matmul(divs.reshape(-1,1),dif.reshape(1,-1))
    interm_steps = []
    # pdb.set_trace()
    for u in divs:
        interm_steps.append(robot.interpolate(m_a,m_b,u))
    interm_steps = np.array(interm_steps)
    return interm_steps

def heuristic_penalized_trelis_solution(selected_points,ik_sols,robot_name,robot,world,penalty = 5):

    coarse_space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.1,
                                        ignoreCollisions = [(robot.link(2),robot.link(3)),
                                                        (robot.link(8),robot.link(6))])
    distances = np.zeros((selected_points.shape[0]+1,selected_points.shape[0]+1))
    for i in range(selected_points.shape[0]):
        point = selected_points[i]
        distances[1:,i+1] = 1000*np.linalg.norm(selected_points-point,axis = 1)
    euc_distances = distances
    tour = runTSP(euc_distances, '/{}_currTSP'.format(robot_name)) 
    tour = (np.array(tour[1:])-1).tolist()
    trelis_ik = np.array(ik_sols)[tour]
    best_indices = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    best_cost = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    for i in tqdm(range(trelis_ik.shape[0]-1)):
        current_iks = trelis_ik[i]
        next_iks = trelis_ik[i+1]
        approx_dist = np.zeros((trelis_ik.shape[1],trelis_ik.shape[1]))
        for j,point in enumerate(next_iks):
            for k,cur_point in enumerate(current_iks):
    #             interpolated_cfigs = interp(cur_point,end,robot)
                if(coarse_space.isVisible(cur_point,point)):
                    approx_dist[j,k] = np.linalg.norm(cur_point-point)
                else:
                    approx_dist[j,k] = penalty*np.linalg.norm(cur_point-point)
    #             approx_dist[j,:] = np.linalg.norm(current_iks-point,axis = 1)

        best_idx = np.argmin(approx_dist,axis = 1)
        cost = np.min(approx_dist, axis = 1)
        if(i != 0):
    #         print(best_cost[i-1,best_idx])
            cost += best_cost[i-1,best_idx]
        best_indices[i,:] = best_idx
        best_cost[i,:] = cost
    j = best_indices.shape[0]-1
    best_idx_path = []
    best_idx_path.append(np.argmin(best_cost[j]).astype(int))
    while(j>=0):
        best_idx_path.append(best_indices[j,best_idx_path[-1]].astype(int))
        j -= 1
    best_idx_path.reverse()
    configuration_sequence = []
    for i,idx in enumerate(best_idx_path):
        configuration_sequence.append(trelis_ik[i,idx])
    return configuration_sequence,trelis_ik

def heuristic_ee_trelis_solution(selected_points,ik_sols,robot_name,robot,world,lamp,penalty = 5):

    coarse_space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.5,
                                        ignoreCollisions = [(robot.link(2),robot.link(3)),
                                                        (robot.link(8),robot.link(6))])
    distances = np.zeros((selected_points.shape[0]+1,selected_points.shape[0]+1))
    for i in range(selected_points.shape[0]):
        point = selected_points[i]
        distances[1:,i+1] = 1000*np.linalg.norm(selected_points-point,axis = 1)
    euc_distances = distances
    tour = runTSP(euc_distances, '/{}_currTSP'.format(robot_name)) 
    tour = (np.array(tour[1:])-1).tolist()
    trelis_ik = np.array(ik_sols)[tour]
    best_indices = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    best_cost = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    for i in tqdm(range(trelis_ik.shape[0]-1)):
        current_iks = trelis_ik[i]
        next_iks = trelis_ik[i+1]
        approx_dist = np.zeros((trelis_ik.shape[1],trelis_ik.shape[1]))
        for j,point in enumerate(next_iks):
            for k,cur_point in enumerate(current_iks):
    #             interpolated_cfigs = interp(cur_point,end,robot)
                if(coarse_space.isVisible(cur_point,point)):
                    approx_dist[j,k] = compute_actual_distance(cur_point,point,lamp,robot)
                else:
                    approx_dist[j,k] = penalty*compute_actual_distance(cur_point,point,lamp,robot)
    #             approx_dist[j,:] = np.linalg.norm(current_iks-point,axis = 1)

        best_idx = np.argmin(approx_dist,axis = 1)
        cost = np.min(approx_dist, axis = 1)
        if(i != 0):
    #         print(best_cost[i-1,best_idx])
            cost += best_cost[i-1,best_idx]
        best_indices[i,:] = best_idx
        best_cost[i,:] = cost
    j = best_indices.shape[0]-1
    best_idx_path = []
    best_idx_path.append(np.argmin(best_cost[j]).astype(int))
    while(j>=0):
        best_idx_path.append(best_indices[j,best_idx_path[-1]].astype(int))
        j -= 1
    best_idx_path.reverse()
    configuration_sequence = []
    for i,idx in enumerate(best_idx_path):
        configuration_sequence.append(trelis_ik[i,idx])
    return configuration_sequence,trelis_ik

def PRM_trellis_solution(adjacency_matrix,ik_sols,lamp,robot):
    distances = np.zeros((adjacency_matrix.shape[0]+1,adjacency_matrix.shape[0]+1))
    distances[1:,1:] = adjacency_matrix
    euc_distances = distances
    tour = runTSP(euc_distances, '/{}_currTSP'.format(robot_name)) 
    tour = (np.array(tour[1:])-1).tolist()
    trelis_ik = np.array(ik_sols)[tour]
    best_indices = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    best_cost = np.zeros((trelis_ik.shape[0]-1,trelis_ik.shape[1]))
    for i in tqdm(range(trelis_ik.shape[0]-1)):
        current_iks = trelis_ik[i]
        next_iks = trelis_ik[i+1]
        approx_dist = np.zeros((trelis_ik.shape[1],trelis_ik.shape[1]))
        for j,point in enumerate(next_iks):
            for k,cur_point in enumerate(current_iks):
    #             interpolated_cfigs = interp(cur_point,end,robot)
                if(coarse_space.isVisible(cur_point,point)):
                    approx_dist[j,k] = compute_actual_distance(cur_point,point,lamp,robot)
                else:
                    approx_dist[j,k] = penalty*compute_actual_distance(cur_point,point,lamp,robot)
    #             approx_dist[j,:] = np.linalg.norm(current_iks-point,axis = 1)

        best_idx = np.argmin(approx_dist,axis = 1)
        cost = np.min(approx_dist, axis = 1)
        if(i != 0):
    #         print(best_cost[i-1,best_idx])
            cost += best_cost[i-1,best_idx]
        best_indices[i,:] = best_idx
        best_cost[i,:] = cost
    j = best_indices.shape[0]-1
    best_idx_path = []
    best_idx_path.append(np.argmin(best_cost[j]).astype(int))
    while(j>=0):
        best_idx_path.append(best_indices[j,best_idx_path[-1]].astype(int))
        j -= 1
    best_idx_path.reverse()
    configuration_sequence = []
    for i,idx in enumerate(best_idx_path):
        configuration_sequence.append(trelis_ik[i,idx])
    return configuration_sequence,trelis_ik


def calculate_path_length_from_configuration_sequence(world,robot,lamp,configuration_sequence,ik_sols,init_duration = 5):
    rrt_space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.1,
                                    ignoreCollisions = [(robot.link(2),robot.link(3)),
                                                    (robot.link(8),robot.link(6))])
    final_path = []
    for i in tqdm(range(len(configuration_sequence)-1)):
        curr_cfig = configuration_sequence[i]
        next_cfig = configuration_sequence[i+1]
        if(rrt_space.isVisible(curr_cfig,next_cfig)):
            path = [curr_cfig,next_cfig]
        else:
            planner = cspace.MotionPlan(rrt_space,type="sbl",connectionThreshold=0.5,bidirectional = 1,shortcut = True,restart = True, knn = 60)  #accepts keyword arguments
            planner.setEndpoints(curr_cfig,next_cfig)
            increment = 500               #there is a little overhead for each planMore call, so it is best not to set the increment too low
            t0 = time.time()
            tmax = init_duration
            no_path = True
            while time.time() - t0 < tmax or no_path:   #max 20 seconds of planning
                planner.planMore(increment)
                path = planner.getPath()
                if(path is None):
                    if(time.time() - t0 > tmax):
                        tmax = 2*tmax

                        print('hmmmm planning failed, extending time by 5 seconds')
                        if(time.time()-t0 > 15):
                            t0 = time.time()
                            tmax = init_duration*2
                            next_cfig = choice(ik_sols[i+1])
                            configuration_sequence[i+1] = next_cfig
                            planner.close()
                            planner = cspace.MotionPlan(rrt_space,type="sbl",connectionThreshold=0.5,bidirectional = 1,shortcut = True,restart = True,knn = 60)  #accepts keyword arguments
                            planner.setEndpoints(curr_cfig,next_cfig)
                else:
                    no_path = False
        if(i != len(configuration_sequence)-1):
    #         print(path)
            final_path.append(path[:-1])
        else:
            final_path.append(path)
        #     if path is not None:
        #         print("Solved, path has",len(path),"milestones")
        #         print("Took time",time.time()-t0)
        #         break

    #     planner.close() 



    appended_path = []
    for i in final_path:
        appended_path.extend(i)
    total_distance = 0 
    planner = cspace.MotionPlan(rrt_space,type="sbl",connectionThreshold=0.5,bidirectional = 1,shortcut = True,restart = True,knn = 60)  #accepts keyword arguments

    for i in range(len(appended_path)-1):
        origin = appended_path[i]
        end = appended_path[i+1]
        total_distance += compute_actual_distance(origin,end,lamp,robot)
    return total_distance,appended_path,final_path