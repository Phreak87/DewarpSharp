
using os;

using sys;

using datetime;

using cv2;

using Image = PIL.Image;

using np = numpy;

using scipy.optimize;

using System.Collections.Generic;

using System;

using System.Linq;

public static class page_dewarp {
    
    public static object PAGE_MARGIN_X = 50;
    
    public static object PAGE_MARGIN_Y = 20;
    
    public static object OUTPUT_ZOOM = 1.0;
    
    public static object OUTPUT_DPI = 300;
    
    public static object REMAP_DECIMATE = 16;
    
    public static object ADAPTIVE_WINSZ = 55;
    
    public static object TEXT_MIN_WIDTH = 15;
    
    public static object TEXT_MIN_HEIGHT = 2;
    
    public static object TEXT_MIN_ASPECT = 1.5;
    
    public static object TEXT_MAX_THICKNESS = 10;
    
    public static object EDGE_MAX_OVERLAP = 1.0;
    
    public static object EDGE_MAX_LENGTH = 100.0;
    
    public static object EDGE_ANGLE_COST = 10.0;
    
    public static object EDGE_MAX_ANGLE = 7.5;
    
    public static object RVEC_IDX = slice(0, 3);
    
    public static object TVEC_IDX = slice(3, 6);
    
    public static object CUBIC_IDX = slice(6, 8);
    
    public static object SPAN_MIN_WIDTH = 30;
    
    public static object SPAN_PX_PER_STEP = 20;
    
    public static object FOCAL_LENGTH = 1.2;
    
    public static object DEBUG_LEVEL = 0;
    
    public static object DEBUG_OUTPUT = "file";
    
    public static object WINDOW_NAME = "Dewarp";
    
    public static object CCOLORS = new List<object> {
        Tuple.Create(255, 0, 0),
        Tuple.Create(255, 63, 0),
        Tuple.Create(255, 127, 0),
        Tuple.Create(255, 191, 0),
        Tuple.Create(255, 255, 0),
        Tuple.Create(191, 255, 0),
        Tuple.Create(127, 255, 0),
        Tuple.Create(63, 255, 0),
        Tuple.Create(0, 255, 0),
        Tuple.Create(0, 255, 63),
        Tuple.Create(0, 255, 127),
        Tuple.Create(0, 255, 191),
        Tuple.Create(0, 255, 255),
        Tuple.Create(0, 191, 255),
        Tuple.Create(0, 127, 255),
        Tuple.Create(0, 63, 255),
        Tuple.Create(0, 0, 255),
        Tuple.Create(63, 0, 255),
        Tuple.Create(127, 0, 255),
        Tuple.Create(191, 0, 255),
        Tuple.Create(255, 0, 255),
        Tuple.Create(255, 0, 191),
        Tuple.Create(255, 0, 127),
        Tuple.Create(255, 0, 63)
    };
    
    public static object K = np.array(new List<object> {
        new List<object> {
            FOCAL_LENGTH,
            0,
            0
        },
        new List<object> {
            0,
            FOCAL_LENGTH,
            0
        },
        new List<object> {
            0,
            0,
            1
        }
    }, dtype: np.float32);
    
    public static object debug_show(object name, object step, object text, object display) {
        if (DEBUG_OUTPUT != "screen") {
            var filetext = text.replace(" ", "_");
            var outfile = name + "_debug_" + str(step) + "_" + filetext + ".png";
            cv2.imwrite(outfile, display);
        }
        if (DEBUG_OUTPUT != "file") {
            var image = display.copy();
            var height = image.shape[0];
            cv2.putText(image, text, Tuple.Create(16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Tuple.Create(0, 0, 0), 3, cv2.LINE_AA);
            cv2.putText(image, text, Tuple.Create(16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Tuple.Create(255, 255, 255), 1, cv2.LINE_AA);
            cv2.imshow(WINDOW_NAME, image);
            while (cv2.waitKey(5) < 0) {
            }
        }
    }
    
    public static object round_nearest_multiple(object i, object factor) {
        i = Convert.ToInt32(i);
        var rem = i % factor;
        if (!rem) {
            return i;
        } else {
            return i + factor - rem;
        }
    }
    
    public static object pix2norm(object shape, object pts) {
        var _tup_1 = shape[::2];
        var height = _tup_1.Item1;
        var width = _tup_1.Item2;
        var scl = 2.0 / max(height, width);
        var offset = np.array(new List<object> {
            width,
            height
        }, dtype: pts.dtype).reshape(Tuple.Create(-1, 1, 2)) * 0.5;
        return (pts - offset) * scl;
    }
    
    public static object norm2pix(object shape, object pts, object as_integer) {
        var _tup_1 = shape[::2];
        var height = _tup_1.Item1;
        var width = _tup_1.Item2;
        var scl = max(height, width) * 0.5;
        var offset = np.array(new List<object> {
            0.5 * width,
            0.5 * height
        }, dtype: pts.dtype).reshape(Tuple.Create(-1, 1, 2));
        var rval = pts * scl + offset;
        if (as_integer) {
            return (rval + 0.5).astype(@int);
        } else {
            return rval;
        }
    }
    
    public static object fltp(object point) {
        return tuple(point.astype(@int).flatten());
    }
    
    public static object draw_correspondences(object img, object dstpoints, object projpts) {
        var display = img.copy();
        dstpoints = norm2pix(img.shape, dstpoints, true);
        projpts = norm2pix(img.shape, projpts, true);
        foreach (var _tup_1 in new List<object> {
            Tuple.Create(projpts, Tuple.Create(255, 0, 0)),
            Tuple.Create(dstpoints, Tuple.Create(0, 0, 255))
        }) {
            var pts = _tup_1.Item1;
            var color = _tup_1.Item2;
            foreach (var point in pts) {
                cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA);
            }
        }
        foreach (var _tup_2 in zip(projpts, dstpoints)) {
            var point_a = _tup_2.Item1;
            var point_b = _tup_2.Item2;
            cv2.line(display, fltp(point_a), fltp(point_b), Tuple.Create(255, 255, 255), 1, cv2.LINE_AA);
        }
        return display;
    }
    
    public static object get_default_params(object corners, object ycoords, object xcoords) {
        // page width and height
        var page_width = np.linalg.norm(corners[1] - corners[0]);
        var page_height = np.linalg.norm(corners[-1] - corners[0]);
        var rough_dims = Tuple.Create(page_width, page_height);
        // our initial guess for the cubic has no slope
        var cubic_slopes = new List<object> {
            0.0,
            0.0
        };
        // object points of flat page in 3D coordinates
        var corners_object3d = np.array(new List<object> {
            new List<object> {
                0,
                0,
                0
            },
            new List<object> {
                page_width,
                0,
                0
            },
            new List<object> {
                page_width,
                page_height,
                0
            },
            new List<object> {
                0,
                page_height,
                0
            }
        });
        // estimate rotation and translation from four 2D-to-3D point
        // correspondences
        var _tup_1 = cv2.solvePnP(corners_object3d, corners, K, np.zeros(5));
        var rvec = _tup_1.Item2;
        var tvec = _tup_1.Item3;
        var span_counts = xcoords.Select(xc => xc.Count);
        var params = np.hstack(Tuple.Create(np.array(rvec).flatten(), np.array(tvec).flatten(), np.array(cubic_slopes).flatten(), ycoords.flatten()) + tuple(xcoords));
        return Tuple.Create(rough_dims, span_counts, params);
    }
    
    public static object project_xy(object xy_coords, object pvec) {
        // get cubic polynomial coefficients given
        //
        //  f(0) = 0, f'(0) = alpha
        //  f(1) = 0, f'(1) = beta
        var _tup_1 = tuple(pvec[CUBIC_IDX]);
        var alpha = _tup_1.Item1;
        var beta = _tup_1.Item2;
        var poly = np.array(new List<object> {
            alpha + beta,
            -2 * alpha - beta,
            alpha,
            0
        });
        xy_coords = xy_coords.reshape(Tuple.Create(-1, 2));
        var z_coords = np.polyval(poly, xy_coords[":",0]);
        var objpoints = np.hstack(Tuple.Create(xy_coords, z_coords.reshape(Tuple.Create(-1, 1))));
        var _tup_2 = cv2.projectPoints(objpoints, pvec[RVEC_IDX], pvec[TVEC_IDX], K, np.zeros(5));
        var image_points = _tup_2.Item1;
        return image_points;
    }
    
    public static object project_keypoints(object pvec, object keypoint_index) {
        var xy_coords = pvec[keypoint_index];
        xy_coords[0,":"] = 0;
        return project_xy(xy_coords, pvec);
    }
    
    public static object resize_to_screen(object src, object maxw = 1280, object maxh = 700, object copy = false) {
        var _tup_1 = src.shape[::2];
        var height = _tup_1.Item1;
        var width = _tup_1.Item2;
        var scl_x = float(width) / maxw;
        var scl_y = float(height) / maxh;
        var scl = Convert.ToInt32(np.ceil(max(scl_x, scl_y)));
        if (scl > 1.0) {
            var inv_scl = 1.0 / scl;
            var img = cv2.resize(src, Tuple.Create(0, 0), null, inv_scl, inv_scl, cv2.INTER_AREA);
        } else if (copy) {
            img = src.copy();
        } else {
            img = src;
        }
        return img;
    }
    
    public static object box(object width, object height) {
        return np.ones(Tuple.Create(height, width), dtype: np.uint8);
    }
    
    public static object get_page_extents(object small) {
        var _tup_1 = small.shape[::2];
        var height = _tup_1.Item1;
        var width = _tup_1.Item2;
        var xmin = PAGE_MARGIN_X;
        var ymin = PAGE_MARGIN_Y;
        var xmax = width - PAGE_MARGIN_X;
        var ymax = height - PAGE_MARGIN_Y;
        var page = np.zeros(Tuple.Create(height, width), dtype: np.uint8);
        cv2.rectangle(page, Tuple.Create(xmin, ymin), Tuple.Create(xmax, ymax), Tuple.Create(255, 255, 255), -1);
        var outline = np.array(new List<object> {
            new List<object> {
                xmin,
                ymin
            },
            new List<object> {
                xmin,
                ymax
            },
            new List<object> {
                xmax,
                ymax
            },
            new List<object> {
                xmax,
                ymin
            }
        });
        return Tuple.Create(page, outline);
    }
    
    public static object get_mask(object name, object small, object pagemask, object masktype) {
        object mask;
        var sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY);
        if (masktype == "text") {
            mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_WINSZ, 25);
            if (DEBUG_LEVEL >= 3) {
                debug_show(name, 0.1, "thresholded", mask);
            }
            mask = cv2.dilate(mask, box(9, 1));
            if (DEBUG_LEVEL >= 3) {
                debug_show(name, 0.2, "dilated", mask);
            }
            mask = cv2.erode(mask, box(1, 3));
            if (DEBUG_LEVEL >= 3) {
                debug_show(name, 0.3, "eroded", mask);
            }
        } else {
            mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_WINSZ, 7);
            if (DEBUG_LEVEL >= 3) {
                debug_show(name, 0.4, "thresholded", mask);
            }
            mask = cv2.erode(mask, box(3, 1), iterations: 3);
            if (DEBUG_LEVEL >= 3) {
                debug_show(name, 0.5, "eroded", mask);
            }
            mask = cv2.dilate(mask, box(8, 2));
            if (DEBUG_LEVEL >= 3) {
                debug_show(name, 0.6, "dilated", mask);
            }
        }
        return np.minimum(mask, pagemask);
    }
    
    public static object interval_measure_overlap(object int_a, object int_b) {
        return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0]);
    }
    
    public static object angle_dist(object angle_b, object angle_a) {
        var diff = angle_b - angle_a;
        while (diff > np.pi) {
            diff -= 2 * np.pi;
        }
        while (diff < -np.pi) {
            diff += 2 * np.pi;
        }
        return np.abs(diff);
    }
    
    public static object blob_mean_and_tangent(object contour) {
        var moments = cv2.moments(contour);
        var area = moments["m00"];
        var mean_x = moments["m10"] / area;
        var mean_y = moments["m01"] / area;
        var moments_matrix = np.array(new List<object> {
            new List<object> {
                moments["mu20"],
                moments["mu11"]
            },
            new List<object> {
                moments["mu11"],
                moments["mu02"]
            }
        }) / area;
        var _tup_1 = cv2.SVDecomp(moments_matrix);
        var svd_u = _tup_1.Item2;
        var center = np.array(new List<object> {
            mean_x,
            mean_y
        });
        var tangent = svd_u[":",0].flatten().copy();
        return Tuple.Create(center, tangent);
    }
    
    public class ContourInfo
        : object {
        
        public ContourInfo(object contour, object rect, object mask) {
            this.contour = contour;
            this.rect = rect;
            this.mask = mask;
            var _tup_1 = blob_mean_and_tangent(contour);
            this.center = _tup_1.Item1;
            this.tangent = _tup_1.Item2;
            this.angle = np.arctan2(this.tangent[1], this.tangent[0]);
            var clx = contour.Select(point => this.proj_x(point));
            var lxmin = min(clx);
            var lxmax = max(clx);
            this.local_xrng = Tuple.Create(lxmin, lxmax);
            this.point0 = this.center + this.tangent * lxmin;
            this.point1 = this.center + this.tangent * lxmax;
            this.pred = null;
            this.succ = null;
        }
        
        public virtual object proj_x(object point) {
            return np.dot(this.tangent, point.flatten() - this.center);
        }
        
        public virtual object local_overlap(object other) {
            var xmin = this.proj_x(other.point0);
            var xmax = this.proj_x(other.point1);
            return interval_measure_overlap(this.local_xrng, Tuple.Create(xmin, xmax));
        }
    }
    
    public static object generate_candidate_edge(object cinfo_a, object cinfo_b) {
        // we want a left of b (so a's successor will be b and b's
        // predecessor will be a) make sure right endpoint of b is to the
        // right of left endpoint of a.
        if (cinfo_a.point0[0] > cinfo_b.point1[0]) {
            var tmp = cinfo_a;
            cinfo_a = cinfo_b;
            cinfo_b = tmp;
        }
        var x_overlap_a = cinfo_a.local_overlap(cinfo_b);
        var x_overlap_b = cinfo_b.local_overlap(cinfo_a);
        var overall_tangent = cinfo_b.center - cinfo_a.center;
        var overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0]);
        var delta_angle = max(angle_dist(cinfo_a.angle, overall_angle), angle_dist(cinfo_b.angle, overall_angle)) * 180 / np.pi;
        // we want the largest overlap in x to be small
        var x_overlap = max(x_overlap_a, x_overlap_b);
        var dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1);
        if (dist > EDGE_MAX_LENGTH || x_overlap > EDGE_MAX_OVERLAP || delta_angle > EDGE_MAX_ANGLE) {
            return null;
        } else {
            var score = dist + delta_angle * EDGE_ANGLE_COST;
            return Tuple.Create(score, cinfo_a, cinfo_b);
        }
    }
    
    public static object make_tight_mask(
        object contour,
        object xmin,
        object ymin,
        object width,
        object height) {
        var tight_mask = np.zeros(Tuple.Create(height, width), dtype: np.uint8);
        var tight_contour = contour - np.array(Tuple.Create(xmin, ymin)).reshape(Tuple.Create(-1, 1, 2));
        cv2.drawContours(tight_mask, new List<object> {
            tight_contour
        }, 0, Tuple.Create(1, 1, 1), -1);
        return tight_mask;
    }
    
    public static object get_contours(object name, object small, object pagemask, object masktype) {
        var mask = get_mask(name, small, pagemask, masktype);
        var _tup_1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE);
        var contours = _tup_1.Item2;
        var contours_out = new List<object>();
        foreach (var contour in contours) {
            var rect = cv2.boundingRect(contour);
            var _tup_2 = rect;
            var xmin = _tup_2.Item1;
            var ymin = _tup_2.Item2;
            var width = _tup_2.Item3;
            var height = _tup_2.Item4;
            if (width < TEXT_MIN_WIDTH || height < TEXT_MIN_HEIGHT || width < TEXT_MIN_ASPECT * height) {
                continue;
            }
            var tight_mask = make_tight_mask(contour, xmin, ymin, width, height);
            if (tight_mask.sum(axis: 0).max() > TEXT_MAX_THICKNESS) {
                continue;
            }
            contours_out.append(ContourInfo(contour, rect, tight_mask));
        }
        if (DEBUG_LEVEL >= 2) {
            visualize_contours(name, small, contours_out);
        }
        return contours_out;
    }
    
    public static object assemble_spans(object name, object small, object pagemask, object cinfo_list) {
        // sort list
        cinfo_list = cinfo_list.OrderBy(cinfo => cinfo.rect[1]).ToList();
        // generate all candidate edges
        var candidate_edges = new List<object>();
        foreach (var _tup_1 in cinfo_list.Select((_p_2,_p_3) => Tuple.Create(_p_3, _p_2))) {
            var i = _tup_1.Item1;
            var cinfo_i = _tup_1.Item2;
            foreach (var j in range(i)) {
                // note e is of the form (score, left_cinfo, right_cinfo)
                var edge = generate_candidate_edge(cinfo_i, cinfo_list[j]);
                if (edge != null) {
                    candidate_edges.append(edge);
                }
            }
        }
        // sort candidate edges by score (lower is better)
        candidate_edges.sort();
        // for each candidate edge
        foreach (var _tup_2 in candidate_edges) {
            var cinfo_a = _tup_2.Item2;
            var cinfo_b = _tup_2.Item3;
            // if left and right are unassigned, join them
            if (cinfo_a.succ == null && cinfo_b.pred == null) {
                cinfo_a.succ = cinfo_b;
                cinfo_b.pred = cinfo_a;
            }
        }
        // generate list of spans as output
        var spans = new List<object>();
        // until we have removed everything from the list
        while (cinfo_list) {
            // get the first on the list
            var cinfo = cinfo_list[0];
            // keep following predecessors until none exists
            while (cinfo.pred) {
                cinfo = cinfo.pred;
            }
            // start a new span
            var cur_span = new List<object>();
            var width = 0.0;
            // follow successors til end of span
            while (cinfo) {
                // remove from list (sadly making this loop *also* O(n^2)
                cinfo_list.remove(cinfo);
                // add to span
                cur_span.append(cinfo);
                width += cinfo.local_xrng[1] - cinfo.local_xrng[0];
                // set successor
                cinfo = cinfo.succ;
            }
            // add if long enough
            if (width > SPAN_MIN_WIDTH) {
                spans.append(cur_span);
            }
        }
        if (DEBUG_LEVEL >= 2) {
            visualize_spans(name, small, pagemask, spans);
        }
        return spans;
    }
    
    public static object sample_spans(object shape, object spans) {
        var span_points = new List<object>();
        foreach (var span in spans) {
            var contour_points = new List<object>();
            foreach (var cinfo in span) {
                var yvals = np.arange(cinfo.mask.shape[0]).reshape(Tuple.Create(-1, 1));
                var totals = (yvals * cinfo.mask).sum(axis: 0);
                var means = totals / cinfo.mask.sum(axis: 0);
                var _tup_1 = cinfo.rect[::2];
                var xmin = _tup_1.Item1;
                var ymin = _tup_1.Item2;
                var step = SPAN_PX_PER_STEP;
                var start = (means.Count - 1) % step / 2;
                contour_points += range(start, means.Count, step).Select(x => Tuple.Create(x + xmin, means[x] + ymin));
            }
            contour_points = np.array(contour_points, dtype: np.float32).reshape(Tuple.Create(-1, 1, 2));
            contour_points = pix2norm(shape, contour_points);
            span_points.append(contour_points);
        }
        return span_points;
    }
    
    public static object keypoints_from_samples(
        object name,
        object small,
        object pagemask,
        object page_outline,
        object span_points) {
        object evec;
        var all_evecs = np.array(new List<object> {
            new List<object> {
                0.0,
                0.0
            }
        });
        var all_weights = 0;
        foreach (var points in span_points) {
            var _tup_1 = cv2.PCACompute(points.reshape(Tuple.Create(-1, 2)), null, maxComponents: 1);
            evec = _tup_1.Item2;
            var weight = np.linalg.norm(points[-1] - points[0]);
            all_evecs += evec * weight;
            all_weights += weight;
        }
        evec = all_evecs / all_weights;
        var x_dir = evec.flatten();
        if (x_dir[0] < 0) {
            x_dir = -x_dir;
        }
        var y_dir = np.array(new List<object> {
            -x_dir[1],
            x_dir[0]
        });
        var pagecoords = cv2.convexHull(page_outline);
        pagecoords = pix2norm(pagemask.shape, pagecoords.reshape(Tuple.Create(-1, 1, 2)));
        pagecoords = pagecoords.reshape(Tuple.Create(-1, 2));
        var px_coords = np.dot(pagecoords, x_dir);
        var py_coords = np.dot(pagecoords, y_dir);
        var px0 = px_coords.min();
        var px1 = px_coords.max();
        var py0 = py_coords.min();
        var py1 = py_coords.max();
        var p00 = px0 * x_dir + py0 * y_dir;
        var p10 = px1 * x_dir + py0 * y_dir;
        var p11 = px1 * x_dir + py1 * y_dir;
        var p01 = px0 * x_dir + py1 * y_dir;
        var corners = np.vstack(Tuple.Create(p00, p10, p11, p01)).reshape(Tuple.Create(-1, 1, 2));
        var ycoords = new List<object>();
        var xcoords = new List<object>();
        foreach (var points in span_points) {
            var pts = points.reshape(Tuple.Create(-1, 2));
            px_coords = np.dot(pts, x_dir);
            py_coords = np.dot(pts, y_dir);
            ycoords.append(py_coords.mean() - py0);
            xcoords.append(px_coords - px0);
        }
        if (DEBUG_LEVEL >= 2) {
            visualize_span_points(name, small, span_points, corners);
        }
        return Tuple.Create(corners, np.array(ycoords), xcoords);
    }
    
    public static object visualize_contours(object name, object small, object cinfo_list) {
        object cinfo;
        object j;
        var regions = np.zeros_like(small);
        foreach (var _tup_1 in cinfo_list.Select((_p_1,_p_2) => Tuple.Create(_p_2, _p_1))) {
            j = _tup_1.Item1;
            cinfo = _tup_1.Item2;
            cv2.drawContours(regions, new List<object> {
                cinfo.contour
            }, 0, CCOLORS[j % CCOLORS.Count], -1);
        }
        var mask = regions.max(axis: 2) != 0;
        var display = small.copy();
        display[mask] = display[mask] / 2 + regions[mask] / 2;
        foreach (var _tup_2 in cinfo_list.Select((_p_3,_p_4) => Tuple.Create(_p_4, _p_3))) {
            j = _tup_2.Item1;
            cinfo = _tup_2.Item2;
            var color = CCOLORS[j % CCOLORS.Count];
            color = tuple(color.Select(c => c / 4));
            cv2.circle(display, fltp(cinfo.center), 3, Tuple.Create(255, 255, 255), 1, cv2.LINE_AA);
            cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1), Tuple.Create(255, 255, 255), 1, cv2.LINE_AA);
        }
        debug_show(name, 1, "contours", display);
    }
    
    public static object visualize_spans(object name, object small, object pagemask, object spans) {
        var regions = np.zeros_like(small);
        foreach (var _tup_1 in spans.Select((_p_1,_p_2) => Tuple.Create(_p_2, _p_1))) {
            var i = _tup_1.Item1;
            var span = _tup_1.Item2;
            var contours = span.Select(cinfo => cinfo.contour);
            cv2.drawContours(regions, contours, -1, CCOLORS[i * 3 % CCOLORS.Count], -1);
        }
        var mask = regions.max(axis: 2) != 0;
        var display = small.copy();
        display[mask] = display[mask] / 2 + regions[mask] / 2;
        display[pagemask == 0] /= 4;
        debug_show(name, 2, "spans", display);
    }
    
    public static object visualize_span_points(object name, object small, object span_points, object corners) {
        var display = small.copy();
        foreach (var _tup_1 in span_points.Select((_p_1,_p_2) => Tuple.Create(_p_2, _p_1))) {
            var i = _tup_1.Item1;
            var points = _tup_1.Item2;
            points = norm2pix(small.shape, points, false);
            var _tup_2 = cv2.PCACompute(points.reshape(Tuple.Create(-1, 2)), null, maxComponents: 1);
            var mean = _tup_2.Item1;
            var small_evec = _tup_2.Item2;
            var dps = np.dot(points.reshape(Tuple.Create(-1, 2)), small_evec.reshape(Tuple.Create(2, 1)));
            var dpm = np.dot(mean.flatten(), small_evec.flatten());
            var point0 = mean + small_evec * (dps.min() - dpm);
            var point1 = mean + small_evec * (dps.max() - dpm);
            foreach (var point in points) {
                cv2.circle(display, fltp(point), 3, CCOLORS[i % CCOLORS.Count], -1, cv2.LINE_AA);
            }
            cv2.line(display, fltp(point0), fltp(point1), Tuple.Create(255, 255, 255), 1, cv2.LINE_AA);
        }
        cv2.polylines(display, new List<object> {
            norm2pix(small.shape, corners, true)
        }, true, Tuple.Create(255, 255, 255));
        debug_show(name, 3, "span points", display);
    }
    
    public static object imgsize(object img) {
        var _tup_1 = img.shape[::2];
        var height = _tup_1.Item1;
        var width = _tup_1.Item2;
        return "{}x{}".format(width, height);
    }
    
    public static object make_keypoint_index(object span_counts) {
        var nspans = span_counts.Count;
        var npts = span_counts.Sum();
        var keypoint_index = np.zeros(Tuple.Create(npts + 1, 2), dtype: @int);
        var start = 1;
        foreach (var _tup_1 in span_counts.Select((_p_1,_p_2) => Tuple.Create(_p_2, _p_1))) {
            var i = _tup_1.Item1;
            var count = _tup_1.Item2;
            var end = start + count;
            keypoint_index[start::(start  +  end),1] = 8 + i;
            start = end;
        }
        keypoint_index[1,0] = np.arange(npts) + 8 + nspans;
        return keypoint_index;
    }
    
    public static object optimize_params(
        object name,
        object small,
        object dstpoints,
        object span_counts,
        object params) {
        object display;
        object projpts;
        var keypoint_index = make_keypoint_index(span_counts);
        Func<object, object> objective = pvec => {
            var ppts = project_keypoints(pvec, keypoint_index);
            return np.sum(Math.Pow(dstpoints - ppts, 2));
        };
        Console.WriteLine("  initial objective is", objective(params));
        if (DEBUG_LEVEL >= 1) {
            projpts = project_keypoints(params, keypoint_index);
            display = draw_correspondences(small, dstpoints, projpts);
            debug_show(name, 4, "keypoints before", display);
        }
        Console.WriteLine("  optimizing", params.Count, "parameters...");
        var start = datetime.datetime.now();
        var res = scipy.optimize.minimize(objective, params, method: "Powell");
        var end = datetime.datetime.now();
        Console.WriteLine("  optimization took", round((end - start).total_seconds(), 2), "sec.");
        Console.WriteLine("  final objective is", res.fun);
        params = res.x;
        if (DEBUG_LEVEL >= 1) {
            projpts = project_keypoints(params, keypoint_index);
            display = draw_correspondences(small, dstpoints, projpts);
            debug_show(name, 5, "keypoints after", display);
        }
        return params;
    }
    
    public static object get_page_dims(object corners, object rough_dims, object params) {
        var dst_br = corners[2].flatten();
        var dims = np.array(rough_dims);
        Func<object, object> objective = dims => {
            var proj_br = project_xy(dims, params);
            return np.sum(Math.Pow(dst_br - proj_br.flatten(), 2));
        };
        var res = scipy.optimize.minimize(objective, dims, method: "Powell");
        dims = res.x;
        Console.WriteLine("  got page dims", dims[0], "x", dims[1]);
        return dims;
    }
    
    public static object remap_image(
        object name,
        object img,
        object small,
        object page_dims,
        object params) {
        var height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0];
        height = round_nearest_multiple(height, REMAP_DECIMATE);
        var width = round_nearest_multiple(height * page_dims[0] / page_dims[1], REMAP_DECIMATE);
        Console.WriteLine("  output will be {}x{}".format(width, height));
        var height_small = height / REMAP_DECIMATE;
        var width_small = width / REMAP_DECIMATE;
        var page_x_range = np.linspace(0, page_dims[0], width_small);
        var page_y_range = np.linspace(0, page_dims[1], height_small);
        var _tup_1 = np.meshgrid(page_x_range, page_y_range);
        var page_x_coords = _tup_1.Item1;
        var page_y_coords = _tup_1.Item2;
        var page_xy_coords = np.hstack(Tuple.Create(page_x_coords.flatten().reshape(Tuple.Create(-1, 1)), page_y_coords.flatten().reshape(Tuple.Create(-1, 1))));
        page_xy_coords = page_xy_coords.astype(np.float32);
        var image_points = project_xy(page_xy_coords, params);
        image_points = norm2pix(img.shape, image_points, false);
        var image_x_coords = image_points[":",0,0].reshape(page_x_coords.shape);
        var image_y_coords = image_points[":",0,1].reshape(page_y_coords.shape);
        image_x_coords = cv2.resize(image_x_coords, Tuple.Create(width, height), interpolation: cv2.INTER_CUBIC);
        image_y_coords = cv2.resize(image_y_coords, Tuple.Create(width, height), interpolation: cv2.INTER_CUBIC);
        var img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
        var remapped = cv2.remap(img_gray, image_x_coords, image_y_coords, cv2.INTER_CUBIC, null, cv2.BORDER_REPLICATE);
        var thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADAPTIVE_WINSZ, 25);
        var pil_image = Image.fromarray(thresh);
        pil_image = pil_image.convert("1");
        var threshfile = name + "_thresh.png";
        pil_image.save(threshfile, dpi: Tuple.Create(OUTPUT_DPI, OUTPUT_DPI));
        if (DEBUG_LEVEL >= 1) {
            height = small.shape[0];
            width = Convert.ToInt32(round(height * float(thresh.shape[1]) / thresh.shape[0]));
            var display = cv2.resize(thresh, Tuple.Create(width, height), interpolation: cv2.INTER_AREA);
            debug_show(name, 6, "output", display);
        }
        return threshfile;
    }
    
    public static object main() {
        if (sys.argv.Count < 2) {
            Console.WriteLine("usage:", sys.argv[0], "IMAGE1 [IMAGE2 ...]");
            sys.exit(0);
        }
        if (DEBUG_LEVEL > 0 && DEBUG_OUTPUT != "file") {
            cv2.namedWindow(WINDOW_NAME);
        }
        var outfiles = new List<object>();
        foreach (var imgfile in sys.argv[1]) {
            var img = cv2.imread(imgfile);
            var small = resize_to_screen(img);
            var basename = os.path.basename(imgfile);
            var _tup_1 = os.path.splitext(basename);
            var name = _tup_1.Item1;
            Console.WriteLine("loaded", basename, "with size", imgsize(img));
            Console.WriteLine("and resized to", imgsize(small));
            if (DEBUG_LEVEL >= 3) {
                debug_show(name, 0.0, "original", small);
            }
            var _tup_2 = get_page_extents(small);
            var pagemask = _tup_2.Item1;
            var page_outline = _tup_2.Item2;
            var cinfo_list = get_contours(name, small, pagemask, "text");
            var spans = assemble_spans(name, small, pagemask, cinfo_list);
            if (spans.Count < 3) {
                Console.WriteLine("  detecting lines because only", spans.Count, "text spans");
                cinfo_list = get_contours(name, small, pagemask, "line");
                var spans2 = assemble_spans(name, small, pagemask, cinfo_list);
                if (spans2.Count > spans.Count) {
                    spans = spans2;
                }
            }
            if (spans.Count < 1) {
                Console.WriteLine("skipping", name, "because only", spans.Count, "spans");
                continue;
            }
            var span_points = sample_spans(small.shape, spans);
            Console.WriteLine("  got", spans.Count, "spans");
            Console.WriteLine("with", span_points.Select(pts => pts.Count).Sum(), "points.");
            var _tup_3 = keypoints_from_samples(name, small, pagemask, page_outline, span_points);
            var corners = _tup_3.Item1;
            var ycoords = _tup_3.Item2;
            var xcoords = _tup_3.Item3;
            var _tup_4 = get_default_params(corners, ycoords, xcoords);
            var rough_dims = _tup_4.Item1;
            var span_counts = _tup_4.Item2;
            var params = _tup_4.Item3;
            var dstpoints = np.vstack(Tuple.Create(corners[0].reshape(Tuple.Create(1, 1, 2))) + tuple(span_points));
            params = optimize_params(name, small, dstpoints, span_counts, params);
            var page_dims = get_page_dims(corners, rough_dims, params);
            var outfile = remap_image(name, img, small, page_dims, params);
            outfiles.append(outfile);
            Console.WriteLine("  wrote", outfile);
            Console.WriteLine();
        }
        Console.WriteLine("to convert to PDF (requires ImageMagick):");
        Console.WriteLine("  convert -compress Group4 " + " ".join(outfiles) + " output.pdf");
    }
    
    static page_dewarp() {
        main();
    }
}
