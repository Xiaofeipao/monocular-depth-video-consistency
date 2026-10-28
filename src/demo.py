"""Point-cloud export and RGB/depth/click-measurement application."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import measure_point,write_ply
from .io import load_run,read_bgr,write_image,write_json
from .visualization import panel


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--frame",type=int,default=0)
    p.add_argument("--output",required=True)
    p.add_argument("--intrinsics",help="JSON with fx,fy,cx,cy in pixels at saved RGB resolution")
    p.add_argument("--clicks",help="JSON list of {x,y,ground_truth_m?} or {x,y,x2,y2,ground_truth_m?}")
    p.add_argument("--interactive",action="store_true",help="OpenCV GUI; click twice to measure 3D distance, q to quit")
    a = p.parse_args(argv)
    run,output = Path(a.run),Path(a.output)
    meta,records = load_run(run)
    if meta['depth_type'] != 'metric' or meta['alignment'] != 'none':
        p.error("Distance measurement requires native metric depth")
    record = records[a.frame]
    rgb = read_bgr(run/"rgb"/(record['id']+".png"))
    depth = np.load(run/"depth"/(record['id']+".npy"),allow_pickle=False)
    h,w = depth.shape
    if a.intrinsics:
        intrinsics = json.loads(Path(a.intrinsics).read_text(encoding="utf-8"))
        source = "user_calibration_at_saved_resolution"
    elif record.get('focal_px'):
        f = record['focal_px']
        intrinsics = dict(fx=f,fy=f,cx=(w-1)/2,cy=(h-1)/2)
        source = "predicted_focal_assumed_center_and_square_pixels"
    else:
        p.error("Provide --intrinsics; this run has no predicted camera focal length")
    output.mkdir(parents=True,exist_ok=False)
    write_ply(output/"point_cloud.ply",depth,cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB),intrinsics)
    app = rgb.copy()
    points,measurements = [],[]
    def click(x,y,x2=None,y2=None,ground_truth_m=None):
        xyz = measure_point(depth,int(x),int(y),intrinsics)
        if x2 is not None and y2 is not None:
            other = measure_point(depth,int(x2),int(y2),intrinsics)
            value,kind = float(np.linalg.norm(xyz-other)),"two_point_distance"
        else:
            value,kind = float(xyz[2]),"axial_depth"
        cv2.circle(app,(int(x),int(y)),4,(0,255,0),-1)
        cv2.putText(app,f"{value:.2f}m",(max(0,min(int(x)+6,w-85)),max(18,int(y))),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
        result = {"x":x,"y":y,"kind":kind,"value_m":value,"xyz_m":xyz.tolist(),"ray_distance_m":float(np.linalg.norm(xyz))}
        if x2 is not None:
            result.update(x2=x2,y2=y2)
        if ground_truth_m is not None:
            if ground_truth_m <= 0:
                raise ValueError("Ground-truth measurement must be positive")
            result.update(ground_truth_m=ground_truth_m,absolute_error_m=abs(value-ground_truth_m),relative_error=abs(value-ground_truth_m)/ground_truth_m)
        measurements.append(result)
        return xyz
    if a.clicks:
        for item in json.loads(Path(a.clicks).read_text(encoding="utf-8")):
            click(**item)
    else:
        # A reproducible initial application output; user may add clicks.
        click(w//2,h//2)
    def make_panel():
        return panel(rgb,depth,*meta['visualization_range'],"meter","RGB | depth | measurement",app)
    if a.interactive:
        def callback(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN and x<w and y<h:
                try:
                    xyz = click(x,y)
                    points.append(xyz)
                    if len(points)%2 == 0:
                        distance = float(np.linalg.norm(points[-1]-points[-2]))
                        print(f"Two-point distance: {distance:.4f} m")
                        measurements.append({"kind":"interactive_two_point_distance","value_m":distance})
                    cv2.imshow("Depth measurement",make_panel())
                except ValueError as exc:
                    print(exc)
        cv2.namedWindow("Depth measurement",cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Depth measurement",callback)
        cv2.imshow("Depth measurement",make_panel())
        while cv2.waitKey(30)&255 not in (ord('q'),27):
            if cv2.getWindowProperty("Depth measurement",cv2.WND_PROP_VISIBLE)<1:
                break
        cv2.destroyAllWindows()
    write_image(output/"rgb_depth_application.png",make_panel())
    scored = [m for m in measurements if "absolute_error_m" in m]
    summary = {"intrinsics":intrinsics,"intrinsics_source":source,"measurements":measurements}
    if scored:
        summary['metrics'] = {"n":len(scored),"MAE_m":float(np.mean([m['absolute_error_m'] for m in scored])),
            "median_AE_m":float(np.median([m['absolute_error_m'] for m in scored])),
            "mean_relative_error":float(np.mean([m['relative_error'] for m in scored]))}
    write_json(output/"measurements.json",summary)
    print(f"Saved application output and point cloud to {output}")


if __name__ == "__main__":
    main()
