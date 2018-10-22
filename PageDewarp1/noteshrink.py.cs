
using @print_function = @@__future__.print_function;

using sys;

using os;

using re;

using subprocess;

using shlex;

using ArgumentParser = argparse.ArgumentParser;

using np = numpy;

using Image = PIL.Image;

using kmeans = scipy.cluster.vq.kmeans;

using vq = scipy.cluster.vq.vq;

using System.Diagnostics;

using System.Collections.Generic;

using System;

public static class noteshrink {
    
    static noteshrink() {
        @"Converts sequence of images to compact PDF while removing speckles,
bleedthrough, etc.

";
        main();
    }
    
    //!/usr/bin/env python
    // for some reason pylint complains about members being undefined :(
    // pylint: disable=E1101
    //#####################################################################
    // Reduces the number of bits per channel in the given image.
    public static object quantize(object image, object bits_per_channel = null) {
        if (bits_per_channel == null) {
            bits_per_channel = 6;
        }
        Debug.Assert(image.dtype == np.uint8);
        var shift = 8 - bits_per_channel;
        var halfbin = 1 << shift >> 1;
        return (image.astype(@int) >> shift << shift) + halfbin;
    }
    
    //#####################################################################
    // Packs a 24-bit RGB triples into a single integer,
    // works on both arrays and tuples.
    public static object pack_rgb(object rgb) {
        object orig_shape = null;
        if (rgb is np.ndarray) {
            Debug.Assert(rgb.shape[-1] == 3);
            orig_shape = rgb.shape[:: - 1];
        } else {
            Debug.Assert(rgb.Count == 3);
            rgb = np.array(rgb);
        }
        rgb = rgb.astype(@int).reshape(Tuple.Create(-1, 3));
        var packed = rgb[":",0] << 16 | rgb[":",1] << 8 | rgb[":",2];
        if (orig_shape == null) {
            return packed;
        } else {
            return packed.reshape(orig_shape);
        }
    }
    
    //#####################################################################
    // Unpacks a single integer or array of integers into one or more
    // 24-bit RGB values.
    // 
    //     
    public static object unpack_rgb(object packed) {
        object orig_shape = null;
        if (packed is np.ndarray) {
            Debug.Assert(packed.dtype == @int);
            orig_shape = packed.shape;
            packed = packed.reshape(Tuple.Create(-1, 1));
        }
        var rgb = Tuple.Create(packed >> 16 & 255, packed >> 8 & 255, packed & 255);
        if (orig_shape == null) {
            return rgb;
        } else {
            return np.hstack(rgb).reshape(orig_shape + Tuple.Create(3));
        }
    }
    
    //#####################################################################
    // Obtains the background color from an image or array of RGB colors
    // by grouping similar colors into bins and finding the most frequent
    // one.
    // 
    //     
    public static object get_bg_color(object image, object bits_per_channel = null) {
        Debug.Assert(image.shape[-1] == 3);
        var quantized = quantize(image, bits_per_channel).astype(@int);
        var packed = pack_rgb(quantized);
        var _tup_1 = np.unique(packed, return_counts: true);
        var unique = _tup_1.Item1;
        var counts = _tup_1.Item2;
        var packed_mode = unique[counts.argmax()];
        return unpack_rgb(packed_mode);
    }
    
    //#####################################################################
    // Convert an RGB image or array of RGB colors to saturation and
    // value, returning each one as a separate 32-bit floating point array or
    // value.
    // 
    //     
    public static object rgb_to_sv(object rgb) {
        if (!(rgb is np.ndarray)) {
            rgb = np.array(rgb);
        }
        var axis = rgb.shape.Count - 1;
        var cmax = rgb.max(axis: axis).astype(np.float32);
        var cmin = rgb.min(axis: axis).astype(np.float32);
        var delta = cmax - cmin;
        var saturation = delta.astype(np.float32) / cmax.astype(np.float32);
        saturation = np.where(cmax == 0, 0, saturation);
        var value = cmax / 255.0;
        return Tuple.Create(saturation, value);
    }
    
    //#####################################################################
    // Runs the postprocessing command on the file provided.
    public static object postprocess(object output_filename, object options) {
        object result;
        Debug.Assert(options.postprocess_cmd);
        var _tup_1 = os.path.splitext(output_filename);
        var @base = _tup_1.Item1;
        var post_filename = @base + options.postprocess_ext;
        var cmd = options.postprocess_cmd;
        cmd = cmd.replace("%i", output_filename);
        cmd = cmd.replace("%o", post_filename);
        cmd = cmd.replace("%e", options.postprocess_ext);
        var subprocess_args = shlex.split(cmd);
        if (os.path.exists(post_filename)) {
            os.unlink(post_filename);
        }
        if (!options.quiet) {
            Console.WriteLine("  running \"{}\"...".format(cmd), end: " ");
            sys.stdout.flush();
        }
        try {
            result = subprocess.call(subprocess_args);
            var before = os.stat(output_filename).st_size;
            var after = os.stat(post_filename).st_size;
        } catch (OSError) {
            result = -1;
        }
        if (result == 0) {
            if (!options.quiet) {
                Console.WriteLine("{:.1f}% reduction".format(100 * (1.0 - float(after) / before)));
            }
            return post_filename;
        } else {
            sys.stderr.write("warning: postprocessing failed!\n");
            return null;
        }
    }
    
    //#####################################################################
    // Convert a string (i.e. 85) to a fraction (i.e. .85).
    public static object percent(object @string) {
        return float(@string) / 100.0;
    }
    
    //#####################################################################
    // Parse the command-line arguments for this program.
    public static object get_argument_parser() {
        var parser = ArgumentParser(description: "convert scanned, hand-written notes to PDF");
        var show_default = " (default %(default)s)";
        parser.add_argument("filenames", metavar: "IMAGE", nargs: "+", help: "files to convert");
        parser.add_argument("-q", dest: "quiet", action: "store_true", @default: false, help: "reduce program output");
        parser.add_argument("-b", dest: "basename", metavar: "BASENAME", @default: "page", help: "output PNG filename base" + show_default);
        parser.add_argument("-o", dest: "pdfname", metavar: "PDF", @default: "output.pdf", help: "output PDF filename" + show_default);
        parser.add_argument("-v", dest: "value_threshold", metavar: "PERCENT", type: percent, @default: "25", help: "background value threshold %%" + show_default);
        parser.add_argument("-s", dest: "sat_threshold", metavar: "PERCENT", type: percent, @default: "20", help: "background saturation \"threshold %%\"" + show_default);
        parser.add_argument("-n", dest: "num_colors", type: @int, @default: "8", help: "number of output colors " + show_default);
        parser.add_argument("-p", dest: "sample_fraction", metavar: "PERCENT", type: percent, @default: "5", help: "%% of pixels to sample" + show_default);
        parser.add_argument("-w", dest: "white_bg", action: "store_true", @default: false, help: "make background white");
        parser.add_argument("-g", dest: "global_palette", action: "store_true", @default: false, help: "use one global palette for all pages");
        parser.add_argument("-S", dest: "saturate", action: "store_false", @default: true, help: "do not saturate colors");
        parser.add_argument("-K", dest: "sort_numerically", action: "store_false", @default: true, help: "keep filenames ordered as specified; \"use if you *really* want IMG_10.png to \"\"precede IMG_2.png\"");
        parser.add_argument("-P", dest: "postprocess_cmd", @default: null, help: "set postprocessing command (see -O, -C, -Q)");
        parser.add_argument("-e", dest: "postprocess_ext", @default: "_post.png", help: "filename suffix/extension for \"postprocessing command\"");
        parser.add_argument("-O", dest: "postprocess_cmd", action: "store_const", @const: "optipng -silent %i -out %o", help: "same as -P \"%(const)s\"");
        parser.add_argument("-C", dest: "postprocess_cmd", action: "store_const", @const: "pngcrush -q %i %o", help: "same as -P \"%(const)s\"");
        parser.add_argument("-Q", dest: "postprocess_cmd", action: "store_const", @const: "pngquant --ext %e %i", help: "same as -P \"%(const)s\"");
        parser.add_argument("-c", dest: "pdf_cmd", metavar: "COMMAND", @default: "convert %i %o", help: "PDF command (default \"%(default)s\")");
        return parser;
    }
    
    //#####################################################################
    // Get the filenames from the command line, optionally sorted by
    // number, so that IMG_10.png is re-arranged to come after IMG_9.png.
    // This is a nice feature because some scanner programs (like Image
    // Capture on Mac OS X) automatically number files without leading zeros,
    // and this way you can supply files using a wildcard and still have the
    // pages ordered correctly.
    // 
    //     
    public static object get_filenames(object options) {
        object num;
        if (!options.sort_numerically) {
            return options.filenames;
        }
        var filenames = new List<object>();
        foreach (var filename in options.filenames) {
            var basename = os.path.basename(filename);
            var _tup_1 = os.path.splitext(basename);
            var root = _tup_1.Item1;
            var matches = re.findall(@"[0-9]+", root);
            if (matches) {
                num = Convert.ToInt32(matches[-1]);
            } else {
                num = -1;
            }
            filenames.append(Tuple.Create(num, filename));
        }
        return filenames.OrderBy(_p_1 => _p_1).ToList().Select(Tuple.Create(_, fn) => fn);
    }
    
    //#####################################################################
    // Load an image with Pillow and convert it to numpy array. Also
    // returns the image DPI in x and y as a tuple.
    public static object load(object input_filename) {
        object dpi;
        object pil_img;
        try {
            pil_img = Image.open(input_filename);
        } catch (IOError) {
            sys.stderr.write("warning: error opening {}\n".format(input_filename));
            return Tuple.Create(null, null);
        }
        if (pil_img.mode != "RGB") {
            pil_img = pil_img.convert("RGB");
        }
        if (pil_img.info.Contains("dpi")) {
            dpi = pil_img.info["dpi"];
        } else {
            dpi = Tuple.Create(300, 300);
        }
        var img = np.array(pil_img);
        return Tuple.Create(img, dpi);
    }
    
    //#####################################################################
    // Pick a fixed percentage of pixels in the image, returned in random
    // order.
    public static object sample_pixels(object img, object options) {
        var pixels = img.reshape(Tuple.Create(-1, 3));
        var num_pixels = pixels.shape[0];
        var num_samples = Convert.ToInt32(num_pixels * options.sample_fraction);
        var idx = np.arange(num_pixels);
        np.random.shuffle(idx);
        return pixels[idx[::num_samples]];
    }
    
    //#####################################################################
    // Determine whether each pixel in a set of samples is foreground by
    // comparing it to the background color. A pixel is classified as a
    // foreground pixel if either its value or saturation differs from the
    // background by a threshold.
    public static object get_fg_mask(object bg_color, object samples, object options) {
        var _tup_1 = rgb_to_sv(bg_color);
        var s_bg = _tup_1.Item1;
        var v_bg = _tup_1.Item2;
        var _tup_2 = rgb_to_sv(samples);
        var s_samples = _tup_2.Item1;
        var v_samples = _tup_2.Item2;
        var s_diff = np.abs(s_bg - s_samples);
        var v_diff = np.abs(v_bg - v_samples);
        return v_diff >= options.value_threshold | s_diff >= options.sat_threshold;
    }
    
    //#####################################################################
    // Extract the palette for the set of sampled RGB values. The first
    // palette entry is always the background color; the rest are determined
    // from foreground pixels by running K-means clustering. Returns the
    // palette, as well as a mask corresponding to the foreground pixels.
    // 
    //     
    public static object get_palette(object samples, object options, object return_mask = false, object kmeans_iter = 40) {
        if (!options.quiet) {
            Console.WriteLine("  getting palette...");
        }
        var bg_color = get_bg_color(samples, 6);
        var fg_mask = get_fg_mask(bg_color, samples, options);
        var _tup_1 = kmeans(samples[fg_mask].astype(np.float32), options.num_colors - 1, iter: kmeans_iter);
        var centers = _tup_1.Item1;
        var palette = np.vstack(Tuple.Create(bg_color, centers)).astype(np.uint8);
        if (!return_mask) {
            return palette;
        } else {
            return Tuple.Create(palette, fg_mask);
        }
    }
    
    //#####################################################################
    // Apply the pallete to the given image. The first step is to set all
    // background pixels to the background color; then, nearest-neighbor
    // matching is used to map each foreground color to the closest one in
    // the palette.
    // 
    //     
    public static object apply_palette(object img, object palette, object options) {
        if (!options.quiet) {
            Console.WriteLine("  applying palette...");
        }
        var bg_color = palette[0];
        var fg_mask = get_fg_mask(bg_color, img, options);
        var orig_shape = img.shape;
        var pixels = img.reshape(Tuple.Create(-1, 3));
        fg_mask = fg_mask.flatten();
        var num_pixels = pixels.shape[0];
        var labels = np.zeros(num_pixels, dtype: np.uint8);
        var _tup_1 = vq(pixels[fg_mask], palette);
        labels[fg_mask] = _tup_1.Item1;
        return labels.reshape(orig_shape[:: - 1]);
    }
    
    //#####################################################################
    // Save the label/palette pair out as an indexed PNG image.  This
    // optionally saturates the pallete by mapping the smallest color
    // component to zero and the largest one to 255, and also optionally sets
    // the background color to pure white.
    // 
    //     
    public static object save(
        object output_filename,
        object labels,
        object palette,
        object dpi,
        object options) {
        if (!options.quiet) {
            Console.WriteLine("  saving {}...".format(output_filename));
        }
        if (options.saturate) {
            palette = palette.astype(np.float32);
            var pmin = palette.min();
            var pmax = palette.max();
            palette = 255 * (palette - pmin) / (pmax - pmin);
            palette = palette.astype(np.uint8);
        }
        if (options.white_bg) {
            palette = palette.copy();
            palette[0] = Tuple.Create(255, 255, 255);
        }
        var output_img = Image.fromarray(labels, "P");
        output_img.putpalette(palette.flatten());
        output_img.save(output_filename, dpi: dpi);
    }
    
    //#####################################################################
    // Fetch the global palette for a series of input files by merging
    // their samples together into one large array.
    // 
    //     
    public static object get_global_palette(object filenames, object options) {
        var input_filenames = new List<object>();
        var all_samples = new List<object>();
        if (!options.quiet) {
            Console.WriteLine("building global palette...");
        }
        foreach (var input_filename in filenames) {
            var _tup_1 = load(input_filename);
            var img = _tup_1.Item1;
            if (img == null) {
                continue;
            }
            if (!options.quiet) {
                Console.WriteLine("  processing {}...".format(input_filename));
            }
            var samples = sample_pixels(img, options);
            input_filenames.append(input_filename);
            all_samples.append(samples);
        }
        var num_inputs = input_filenames.Count;
        all_samples = all_samples.Select(s => s[::int(round((float(s.shape[0])  / num_inputs)))]);
        all_samples = np.vstack(tuple(all_samples));
        var global_palette = get_palette(all_samples, options);
        if (!options.quiet) {
            Console.WriteLine("  done\n");
        }
        return Tuple.Create(input_filenames, global_palette);
    }
    
    //#####################################################################
    // Runs the PDF conversion command to generate the PDF.
    public static object emit_pdf(object outputs, object options) {
        object result;
        object cmd_print;
        var cmd = options.pdf_cmd;
        cmd = cmd.replace("%o", options.pdfname);
        if (outputs.Count > 2) {
            cmd_print = cmd.replace("%i", " ".join(outputs[::2] + new List<object> {
                "..."
            }));
        } else {
            cmd_print = cmd.replace("%i", " ".join(outputs));
        }
        cmd = cmd.replace("%i", " ".join(outputs));
        if (!options.quiet) {
            Console.WriteLine("running PDF command \"{}\"...".format(cmd_print));
        }
        try {
            result = subprocess.call(shlex.split(cmd));
        } catch (OSError) {
            result = -1;
        }
        if (result == 0) {
            if (!options.quiet) {
                Console.WriteLine("  wrote", options.pdfname);
            }
        } else {
            sys.stderr.write("warning: PDF command failed\n");
        }
    }
    
    //#####################################################################
    // Main function for this program when run as script.
    public static object notescan_main(object options) {
        object palette;
        var filenames = get_filenames(options);
        var outputs = new List<object>();
        var do_global = options.global_palette && filenames.Count > 1;
        if (do_global) {
            var _tup_1 = get_global_palette(filenames, options);
            filenames = _tup_1.Item1;
            palette = _tup_1.Item2;
        }
        var do_postprocess = @bool(options.postprocess_cmd);
        foreach (var input_filename in filenames) {
            var _tup_2 = load(input_filename);
            var img = _tup_2.Item1;
            var dpi = _tup_2.Item2;
            if (img == null) {
                continue;
            }
            var output_filename = "{}{:04d}.png".format(options.basename, outputs.Count);
            if (!options.quiet) {
                Console.WriteLine("opened", input_filename);
            }
            if (!do_global) {
                var samples = sample_pixels(img, options);
                palette = get_palette(samples, options);
            }
            var labels = apply_palette(img, palette, options);
            save(output_filename, labels, palette, dpi, options);
            if (do_postprocess) {
                var post_filename = postprocess(output_filename, options);
                if (post_filename) {
                    output_filename = post_filename;
                } else {
                    do_postprocess = false;
                }
            }
            outputs.append(output_filename);
            if (!options.quiet) {
                Console.WriteLine("  done\n");
            }
        }
        emit_pdf(outputs, options);
    }
    
    //#####################################################################
    // Parse args and call notescan_main().
    public static object main() {
        notescan_main(options: get_argument_parser().parse_args());
    }
}
