import os
import struct
import math
import numpy as np
import json
try:
    import bpy
    import mathutils
except ImportError:
    pass

def process_and_deform_splat(splat_path, output_path, mesh_obj, armature_obj, setup_frame,
                             *, normalization_yaw_deg=0.0):
    if not os.path.exists(splat_path):
        return None

    print(f"[SPLAT] Processing splat: {splat_path}")
    
    # We assume mesh_obj is currently in T-pose (this function should be called before pose is changed)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tpose_bvh = mathutils.bvhtree.BVHTree.FromObject(mesh_obj, depsgraph)
    
    # Get the T-pose mesh to access vertices and vertex groups
    tpose_mesh = mesh_obj.evaluated_get(depsgraph).to_mesh()
    
    # Read the PLY file
    with open(splat_path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode('utf-8')
            header.append(line)
            if line.strip() == "end_header":
                break
        
        # Parse vertex count
        vertex_count = 0
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
        
        data = f.read()
    
    # 17 floats * 4 bytes = 68 bytes per point
    pts = np.frombuffer(data, dtype=np.float32).reshape(-1, 17).copy()
    
    # Apply the same Z-axis yaw rotation that normalize_model_orientation
    # applied to the mesh inside Blender.  Without this, splat positions
    # are in the original (pre-normalization) coordinate frame while the
    # mesh vertices have already been rotated, causing BVH nearest-vertex
    # lookups to match the wrong geometry.
    if abs(normalization_yaw_deg) > 0.01:
        angle_rad = math.radians(normalization_yaw_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        x = pts[:, 0].copy()
        y = pts[:, 1].copy()
        pts[:, 0] = cos_a * x - sin_a * y
        pts[:, 1] = sin_a * x + cos_a * y
        # Also rotate the per-splat quaternions (WXYZ in columns 13-16)
        # by the same Z-axis yaw so their orientations stay consistent.
        q_yaw = mathutils.Quaternion((0, 0, 1), angle_rad)
        for i in range(len(pts)):
            q = mathutils.Quaternion((pts[i, 13], pts[i, 14], pts[i, 15], pts[i, 16]))
            q = q_yaw @ q
            q.normalize()
            pts[i, 13:17] = [q.w, q.x, q.y, q.z]
        print(f"[SPLAT] Pre-rotated splat data by {normalization_yaw_deg:.1f}° around Z")

    # Get vertex groups
    vgroups = {g.index: g.name for g in mesh_obj.vertex_groups}
    
    print("[SPLAT] Finding nearest vertices...")
    splat_vertices = pts[:, :3]
    
    # Save the original pose frame
    orig_frame = bpy.context.scene.frame_current
    
    # Change frame to donor pose
    bpy.context.scene.frame_set(setup_frame)
    bpy.context.view_layer.update()
    
    # Precompute LBS matrices for each bone
    bone_lbs_matrices = {}
    if armature_obj:
        for pbone in armature_obj.pose.bones:
            bind_mat = pbone.bone.matrix_local
            pose_mat = pbone.matrix
            lbs_mat = pose_mat @ bind_mat.inverted()
            bone_lbs_matrices[pbone.name] = np.array(lbs_mat)
    
    print("[SPLAT] Deforming points...")
    splat_weights = {} # Store dominant bone per splat index for segmentation!
    
    for i in range(vertex_count):
        loc = mathutils.Vector(tuple(splat_vertices[i]))
        nearest_loc, normal, face_idx, dist = tpose_bvh.find_nearest(loc)
        if face_idx is None:
            continue
            
        face = tpose_mesh.polygons[face_idx]
        
        # Find nearest vertex in face
        best_v = None
        best_d = float('inf')
        for v_idx in face.vertices:
            v_loc = tpose_mesh.vertices[v_idx].co
            d = (v_loc - loc).length
            if d < best_d:
                best_d = d
                best_v = tpose_mesh.vertices[v_idx]
        
        if not best_v:
            continue
            
        # Get weights
        weights = {}
        dominant_bone = None
        max_w = -1.0
        
        for g in best_v.groups:
            bname = vgroups.get(g.group)
            if bname and bname in bone_lbs_matrices:
                w = g.weight
                weights[bname] = w
                if w > max_w:
                    max_w = w
                    dominant_bone = bname
                    
        if dominant_bone:
            splat_weights[str(i)] = dominant_bone
                
        # Normalize weights
        total_w = sum(weights.values())
        if total_w > 0:
            for k in weights:
                weights[k] /= total_w
                
            # Apply LBS
            final_mat = np.zeros((4, 4), dtype=np.float32)
            for bname, w in weights.items():
                final_mat += bone_lbs_matrices[bname] * w
                
            # Transform position
            pos = np.append(pts[i, :3], 1.0)
            new_pos = final_mat @ pos
            pts[i, :3] = new_pos[:3]
            
            # Transform rotation (quaternion: rot_0=w, rot_1=x, rot_2=y, rot_3=z)
            q = mathutils.Quaternion((pts[i, 13], pts[i, 14], pts[i, 15], pts[i, 16]))
            m = mathutils.Matrix(final_mat.tolist())
            q_rot = m.to_quaternion()
            new_q = q_rot @ q
            new_q.normalize()
            pts[i, 13:17] = [new_q.w, new_q.x, new_q.y, new_q.z]
            
    # Write output PLY
    print(f"[SPLAT] Writing output to {output_path}...")
    with open(output_path, "wb") as f:
        for line in header:
            f.write(line.encode('utf-8'))
        f.write(pts.tobytes())
        
    # Write weights JSON
    weights_path = output_path.replace(".ply", "_weights.json")
    with open(weights_path, "w") as f:
        json.dump(splat_weights, f)
        
    # Restore original frame
    bpy.context.scene.frame_set(orig_frame)
    bpy.context.view_layer.update()
        
    return output_path
