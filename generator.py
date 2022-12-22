#!/usr/bin/env python3

"""
A prototype floor-layout generating application, developed as a part
of the "Semantic-driven graph transformations in floor plan design"
research paper.
"""

import neo4j, cairo

from sys import argv, stdout
from math import ceil
from random import choice, randint, shuffle
from collections import namedtuple
from types import SimpleNamespace

FontExtents = namedtuple('FontExtents', 'ascent descent height max_x_advance max_y_advance')

class GraphGrammar:
    """
    Base class for grammars which transform Neo4j graphs.
    """

    def __init__(self, driver):
        self.driver = driver
        self.db = None

    def dump_graph(self, flog=stdout):
        flog.write('dumping nodes and relationships...\n')
        with self.driver.session() as sess:
            res = sess.run('MATCH (n) RETURN n')
            for r in res:
                flog.write(repr(r[0]))
                flog.write('\n')
            res.consume()
            res = sess.run('MATCH () -[r]-> () RETURN r')
            for r in res:
                flog.write(repr(r[0]))
                flog.write('\n')
            res.consume()
        flog.write('... done\n')

    def query(self, q, *args, **kwargs):
        q = q.strip()
        with self.driver.session() as sess:
            result = sess.run(q, *args, **kwargs)
            records = [ r for r in result ]
            result.consume()
        return records

    def create_start_graph(self):
        self.query('CREATE ()')
        return 'initial vertex created'

    def generate(self, flog=stdout, on_change=None):
        # Find methods which implement grammar rules.
        rules = list()
        for s in dir(self):
            if s.startswith('rule_'):
                m = getattr(self, s, None)
                if callable(m):
                    rules.append( (s[5:], m) )
        if len(rules) == 0:
            raise RuntimeError('this grammar has no rules')
        # Initialize the database.
        self.query('MATCH (n) DETACH DELETE n')
        msg = self.create_start_graph()
        flog.write('start_graph: {}\n'.format(msg))
        if on_change is not None:
            on_change()
        # Apply randomly chosen rules.
        candidates = list(rules)
        while len(candidates) > 0:
            i = randint(0, len(candidates) - 1)
            (name, meth) = candidates[i]
            del candidates[i]
            msg = meth()
            if msg is None:
                flog.write('{}: no match found\n'.format(name))
            else:
                flog.write('{}: {}\n'.format(name, msg))
                if on_change is not None:
                    on_change()
                candidates = list(rules)
        flog.write('done: no rule can be applied\n')


class FloorLayoutGrammar(GraphGrammar):
    """
    Base class providing helper methods used to implement floor-layout
    rules. Layout visualization methods are defined here, too.

    Every rectangular area corresponds to a CP-graph vertex with four
    bonds, which in turn corresponds to five nodes in the database. One
    node represents the area as a whole. It is connected by relations
    named b1 ... b4 to four bond nodes representing its walls, starting
    with the northern wall and proceeding clockwise.

    Bond nodes belonging to two different areas can be connected
    by relations named acc, adj or emb. Which bond is the source and
    which is the target does not matter, because these relations are
    undirected from the semantic point of view.

    Every bond node has attributes storing wall endpoints' locations,
    with x1 <= x2 and y1 <= y2. A standard Cartesian coordinate system
    with X axis pointed rightward and Y axis going up is used. Up is
    north, right is east, etc.
    """

    def __init__(self, driver, layout_width=10, layout_height=10, **kwargs):
        super().__init__(driver)
        self.layout_width = int(layout_width)
        self.layout_height = int(layout_height)
        self.drawing_parameters = SimpleNamespace()
        self.drawing_parameters.__dict__.update(self.default_drawing_parameters.__dict__)
        self.drawing_parameters.__dict__.update(**kwargs)

    def query_match_rect(self, label=None, condition=None):
        """
        Find areas with a given label, which fulfill a given condition.
        Label and/or condition can be left unspecified. If given, the
        condition needs to be a valid Neo4j boolean expression, which
        can use identifiers n b1 b2 b3 b4 to refer to the matched nodes.

        Returns a list of records with five fields: n (which includes
        n.id), llx, lly, urx, ury.
        """

        if label is not None:
            s1 = ':`' + label + '`'
        else:
            s1 = ''
        if condition is not None:
            s2 = 'WHERE ' + condition + '\n'
        else:
            s2 = ''
        return self.query('''
MATCH (n{0}),
    (n) -[:b1]-> (b1:bond), (n) -[:b2]-> (b2:bond),
    (n) -[:b3]-> (b3:bond), (n) -[:b4]-> (b4:bond)
{1}RETURN n, b3.x1, b3.y1, b1.x2, b1.y2
                '''.format(s1, s2))

    def query_delete_rect(self, main_node_id):
        return self.query('''
MATCH (n),
    (n) -[:b1]-> (b1:bond), (n) -[:b2]-> (b2:bond),
    (n) -[:b3]-> (b3:bond), (n) -[:b4]-> (b4:bond)
WHERE ID(n) = {i}
DETACH DELETE n, b1, b2, b3, b4
                ''', i=main_node_id)

    NodeAndBondIds = namedtuple('NodeAndBondIds', ['id', 'b1_id', 'b2_id', 'b3_id', 'b4_id'])

    def query_create_rect(self, label, llx, lly, urx, ury):
        rs = self.query('''
CREATE (n:`{label}` {{ area: ({urx} - {llx}) * ({ury} - {lly}) }}),
    (n) -[:b1]-> (b1:bond {{ x1: {llx}, y1: {ury}, x2: {urx}, y2: {ury} }}),
    (n) -[:b2]-> (b2:bond {{ x1: {urx}, y1: {lly}, x2: {urx}, y2: {ury} }}),
    (n) -[:b3]-> (b3:bond {{ x1: {llx}, y1: {lly}, x2: {urx}, y2: {lly} }}),
    (n) -[:b4]-> (b4:bond {{ x1: {llx}, y1: {lly}, x2: {llx}, y2: {ury} }})
RETURN ID(n), ID(b1), ID(b2), ID(b3), ID(b4)
                '''.format(label=label, llx=llx, lly=lly, urx=urx, ury=ury))
        return self.NodeAndBondIds( *(rs[0]) )

    def query_create_edge(self, label, source_id, destination_id):
        self.query('''
MATCH (n), (m)
WHERE ID(n) = {i} AND ID(m) = {j}
CREATE (n) -[:{t}]-> (m)
                '''.format(t=label, i=source_id, j=destination_id))

    def query_embed_eastern(self, old_rect_id, new_rect_id):
        """
        Embed the eastern bond of the newly created sub-area by
        connecting it to selected bonds of the original area's
        context.

        To do that, first find relations connecting the eastern bond
        of the original area to western bonds of other areas (this
        implies that these other areas are geometrically adjacent).
        These bonds are the context. Then, check which context bonds
        are overlapping (in the geometric sense) the eastern bond of
        the new area, and create "emb" relations connecting them to
        this newly created bond.

        This method assumes that the sub-area given as the argument is
        on the eastern side of the original area, it does not verify
        this fact (which could be done by checking if in all bonds
        attributes x1 and x2 have a single, identical value).
        """

        self.query('''
MATCH (old) -[:b2]-> (old2:bond) -- (ctx4:bond),
    (new) -[:b2]-> (new2:bond)
WHERE ID(old) = {0} AND ID(new) = {1} AND
    NOT (new2.y2 <= ctx4.y1 OR new2.y1 >= ctx4.y2)
CREATE (new2) -[:emb]-> (ctx4)
RETURN ID(ctx4)
                '''.format(old_rect_id, new_rect_id))

    def query_embed_western(self, old_rect_id, new_rect_id):
        """See query_embed_eastern()."""

        self.query('''
MATCH (old) -[:b4]-> (old4:bond) -- (ctx2:bond),
    (new) -[:b4]-> (new4:bond)
WHERE ID(old) = {0} AND ID(new) = {1} AND
    NOT (new4.y2 <= ctx2.y1 OR new4.y1 >= ctx2.y2)
CREATE (new4) -[:emb]-> (ctx2)
RETURN ID(ctx2)
                '''.format(old_rect_id, new_rect_id))


    def create_start_graph(self):
        self.query_create_rect('S', 0, 0, self.layout_width, self.layout_height)
        return 'initial area created, size {0} x {1} units'.format(
                    self.layout_width, self.layout_height)

    # --------------------------------------------------------------------------------------------

    default_drawing_parameters = SimpleNamespace(
        layout_unit = 18,   # 18 pt == 1/4 in == 6.35 mm
        line_width = 0.5,
        font_size = 7,
        page_margin = 1,
        room_padding = 0,
        draw_doors = True,
        draw_potential_doors = True,
        door_halfwidth = 6,
        doorjamb_halfdepth = 2,
        draw_edges = False,
        draw_grid = False)

    def save_pdf(self, fname):
        self.__create_pdf_surface(fname)
        self.__draw_page(self.__pdf_context)
        self.__pdf_surface.show_page()
        self.__destroy_pdf_surface()

    def __create_pdf_surface(self, fname):
        u = self.drawing_parameters.layout_unit
        m = self.drawing_parameters.page_margin
        self.__pdf_surface = cairo.PDFSurface(fname, self.layout_width * u + 2 * m, self.layout_height * u + 2 * m)
        self.__pdf_context = self.__make_context(self.__pdf_surface)

    def __destroy_pdf_surface(self):
        del self.__pdf_context
        self.__pdf_surface.finish()
        del self.__pdf_surface

    def save_png(self, fname, ppi=72):
        self.__create_img_surface(ppi)
        self.__draw_page(self.__img_context)
        self.__img_surface.write_to_png(fname)
        self.__destroy_img_surface()

    def __create_img_surface(self, ppi, color=(1, 1, 1)):
        scale = ppi / 72.0
        u = self.drawing_parameters.layout_unit
        m = self.drawing_parameters.page_margin
        self.__img_surface = cairo.ImageSurface(cairo.FORMAT_RGB24,
                ceil(scale * (self.layout_width * u + 2 * m)),
                ceil(scale * (self.layout_height * u + 2 * m)))
        self.__img_context = self.__make_context(self.__img_surface, scale=scale, paint_color=color)

    def __destroy_img_surface(self):
        del self.__img_context
        del self.__img_surface

    def __make_context(self, surface, scale=1.0, paint_color=None):
        ctx = cairo.Context(surface)
        if scale != 1.0:
            ctx.scale(scale, scale)
        if paint_color is not None:
            ctx.set_source_rgb(*paint_color)
            ctx.paint()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(self.drawing_parameters.line_width)
        return ctx

    def generate_and_multipage_save(self, fbase, ppi=72, **kwargs):
        self.__multipage_fbase = fbase
        self.__multipage_ppi = ppi
        self.__multipage_no = 0
        self.__create_pdf_surface(fbase + '.pdf')
        self.generate(on_change=self.__multipage_callback, **kwargs)
        self.__destroy_pdf_surface()

    def __multipage_callback(self):
        self.__draw_page(self.__pdf_context)
        self.__pdf_surface.show_page()
        self.__multipage_no += 1
        fname = self.__multipage_fbase + '-' + str(self.__multipage_no) + '.png'
        self.save_png(fname, self.__multipage_ppi)

    def __draw_page(self, ctx):

        def layout2cairo(x, y):
            p = self.drawing_parameters
            return (p.page_margin + x * p.layout_unit,
                        p.page_margin + (self.layout_height - y) * p.layout_unit)

        # Get rectangle nodes, store them for later processing.
        rooms = { }
        rs = self.query('''
MATCH (b1:bond) <-[:b1]- (n) -[:b3]-> (b3:bond)
RETURN n, b3.x1, b3.y1, b1.x2, b1.y2
                ''')
        for r in rs:
            assert len(r[0].labels) == 1
            assert r[0].id not in rooms
            rooms[ r[0].id ] = SimpleNamespace(
                n = r[0],
                label = [ x for x in r[0].labels ][0],
                llx = r[1],
                lly = r[2],
                urx = r[3],
                ury = r[4])

        # Get bond-bond relations and their associated nodes, calculate and store
        # wall segments shared by adjacent rooms.
        segments_acc = [ ]
        segments_adj = [ ]
        segments_emb = [ ]
        rs = self.query('''
MATCH (n) --> (b:bond) -[e]-> (c:bond) <-- (m)
WHERE NOT 'bond' IN labels(n) AND NOT 'bond' IN labels(m)
RETURN type(e), b, c, n, m
                ''')
        for (rel, b, c, n, m) in rs:
            llx = max(b['x1'], c['x1'])
            lly = max(b['y1'], c['y1'])
            urx = min(b['x2'], c['x2'])
            ury = min(b['y2'], c['y2'])
            assert (llx == urx and lly < ury) or (lly == ury and llx < urx)
            segment = SimpleNamespace(
                rel = rel,
                between = { n.id, m.id },
                llx = llx,
                lly = lly,
                urx = urx,
                ury = ury)
            if rel == 'acc':
                segments_acc.append(segment)
            elif rel == 'adj':
                segments_adj.append(segment)
            elif rel == 'emb':
                segments_emb.append(segment)
            else:
                assert False

        # Draw background grid.
        if self.drawing_parameters.draw_grid:
            ctx.save()
            ctx.set_source_rgb(0.9, 0.9, 0.9)
            for i in range(self.layout_width + 1):
                ctx.move_to( *layout2cairo(i, 0) )
                ctx.line_to( *layout2cairo(i, self.layout_height) )
            for i in range(self.layout_height + 1):
                ctx.move_to( *layout2cairo(0, i) )
                ctx.line_to( *layout2cairo(self.layout_width, i) )
            ctx.stroke()
            ctx.restore()

        def draw_door_list(doors, wdx, wdy, jdx, jdy, offx, offy, north=False):
            for i in range(len(doors)):
                (x, y) = layout2cairo(doors[i].x, doors[i].y)
                if i > 0 or not north:
                    ctx.line_to(x - wdx + offx, y - wdy + offy)
                    ctx.line_to(x - wdx + jdx + offx, y - wdy + jdy + offy)
                ctx.move_to(x + wdx + jdx + offx, y + wdy + jdy + offy)
                ctx.line_to(x + wdx + offx, y + wdy + offy)

        # Draw walls with door openings (if the drawing parameters allow).
        if self.drawing_parameters.draw_doors:
            for room in rooms.values():
                north_doors = [ ]
                east_doors = [ ]
                south_doors = [ ]
                west_doors = [ ]
                for s in segments_acc:
                    if room.n.id in s.between:
                        d = SimpleNamespace(
                                x = (s.llx + s.urx) / 2.0,
                                y = (s.lly + s.ury) / 2.0)
                        if d.y == room.ury:
                            north_doors.append(d)
                        elif d.x == room.urx:
                            east_doors.append(d)
                        elif d.y == room.lly:
                            south_doors.append(d)
                        elif d.x == room.llx:
                            west_doors.append(d)
                        else:
                            assert False
                # Sort door lists in the drawing order.
                north_doors.sort(key=lambda d: d.x)
                east_doors.sort(key=lambda d: -d.y)
                south_doors.sort(key=lambda d: -d.x)
                west_doors.sort(key=lambda d: d.y)
                # Drawing algorithm starts with the northern wall...
                o = self.drawing_parameters.room_padding
                w = self.drawing_parameters.door_halfwidth
                j = self.drawing_parameters.doorjamb_halfdepth
                if len(north_doors) == 0:
                    (x, y) = layout2cairo((room.llx + room.urx) / 2.0, room.ury)
                    ctx.move_to(x, y + o)
                else:
                    draw_door_list(north_doors, w, 0, 0, j, 0, o, north=True)
                (x, y) = layout2cairo(room.urx, room.ury)
                ctx.line_to(x - o, y + o)
                # ... then line continues along the eastern wall, ...
                draw_door_list(east_doors, 0, w, -j, 0, -o, 0)
                (x, y) = layout2cairo(room.urx, room.lly)
                ctx.line_to(x - o, y - o)
                # ... southern, ...
                draw_door_list(south_doors, -w, 0, 0, -j, 0, -o)
                (x, y) = layout2cairo(room.llx, room.lly)
                ctx.line_to(x + o, y - o)
                # ... western, ...
                draw_door_list(west_doors, 0, -w, j, 0, o, 0)
                (x, y) = layout2cairo(room.llx, room.ury)
                ctx.line_to(x + o, y + o)
                # ... and back to northern.
                if len(north_doors) == 0:
                    # 3/4 instead of 1/2, because last line segment needs to overlap the first.
                    (x, y) = layout2cairo((room.llx + 3 * room.urx) / 4.0, room.ury)
                    ctx.line_to(x, y + o)
                else:
                    (x, y) = layout2cairo(north_doors[0].x, north_doors[0].y)
                    ctx.line_to(x - w, y + o)
                    ctx.line_to(x - w, y + j + o)
                ctx.stroke()

        # Draw rhombuses where door openings may be created by the designer.
        if self.drawing_parameters.draw_potential_doors:
            o = self.drawing_parameters.room_padding
            w = self.drawing_parameters.door_halfwidth
            j = self.drawing_parameters.doorjamb_halfdepth
            for s in segments_emb:
                x = (s.llx + s.urx) / 2.0
                y = (s.lly + s.ury) / 2.0
                (x, y) = layout2cairo(x, y)
                if s.lly == s.ury:
                    # horizontal wall segment
                    ctx.move_to(x - w, y - o)
                    ctx.line_to(x, y - j - o)
                    ctx.line_to(x + w, y - o)
                    ctx.line_to(x + w, y + o)
                    ctx.line_to(x, y + j + o)
                    ctx.line_to(x - w, y + o)
                    ctx.close_path()
                else:
                    ctx.move_to(x + o, y - w)
                    ctx.line_to(x + j + o, y)
                    ctx.line_to(x + o, y + w)
                    ctx.line_to(x - o, y + w)
                    ctx.line_to(x - j - o, y)
                    ctx.line_to(x - o, y - w)
                    ctx.close_path()
                # ctx.save()
                # ctx.set_source_rgb(0.7, 0.7, 1.0)
                # ctx.fill_preserve()
                # ctx.restore()
                ctx.stroke()

        # Draw room/area labels using Cairo toy text API, and if the drawing parameters hide doors
        # then draw continuous walls, too.
        ctx.select_font_face('Bitstream Vera Serif', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(self.drawing_parameters.font_size)
        font_ext = FontExtents._make( ctx.font_extents() )
        for r in rooms.values():
            label = r.label
            area = str(r.n['area']) + ' m²'
            # Note: Cairo has the Y axis pointing down, "lower left" and "upper right" corners
            # will be different, data from the graph model needs to be converted!
            (llx, ury) = layout2cairo(r.llx, r.lly)
            (urx, lly) = layout2cairo(r.urx, r.ury)
            if not self.drawing_parameters.draw_doors:
                o = self.drawing_parameters.room_padding
                ctx.rectangle(llx + o, lly + o, urx - llx - 2 * o, ury - lly - 2 * o)
                ctx.stroke()
            te1 = ctx.text_extents(label)
            te2 = ctx.text_extents(area)
            x_text = (llx + urx) / 2.0
            y_text = (lly + ury) / 2.0 - 0.25 * font_ext.height
            ctx.move_to(x_text - te1.x_advance / 2.0, y_text)
            ctx.show_text(label)
            y_text += 1.1 * font_ext.height
            ctx.move_to(x_text - te2.x_advance / 2.0, y_text)
            ctx.show_text(area)

        def draw_relation(seg):
            x = (seg.llx + seg.urx) / 2.0
            y = (seg.lly + seg.ury) / 2.0
            (x, y) = layout2cairo(x, y)
            if seg.lly == seg.ury:
                # horizontal wall segment => vertical line
                ctx.move_to(x, y - 0.3 * self.drawing_parameters.layout_unit)
                ctx.rel_line_to(0, 0.6 * self.drawing_parameters.layout_unit)
            else:
                ctx.move_to(x - 0.3 * self.drawing_parameters.layout_unit, y)
                ctx.rel_line_to(0.6 * self.drawing_parameters.layout_unit, 0)
            ctx.stroke()

        # Draw semi-transparent lines representing relations.
        if self.drawing_parameters.draw_edges:
            ctx.save()
            ctx.set_line_width(0.2 * self.drawing_parameters.layout_unit)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            for s in segments_acc:
                ctx.set_source_rgba(0.0, 1.0, 0.0, 0.5)
                draw_relation(s)
            for s in segments_adj:
                ctx.set_source_rgba(1.0, 0.0, 0.0, 0.5)
                draw_relation(s)
            for s in segments_emb:
                ctx.set_source_rgba(1.0, 1.0, 0.0, 0.5)
                draw_relation(s)
            ctx.restore()


class ExampleGrammar(FloorLayoutGrammar):
    """
    A working example of a floor layout grammar, which implements graph
    rules displayed in Fig. 7 in the paper. All methods follow the same
    sequence of steps:

    1) Find a match for rule's left-hand side and remember it for
    later. In this grammar all rules have a single area as their LHS;
    remembering a single reference to the Neo4j node representing
    this area is sufficient.

    2) Determine dimensions of sub-areas. Do it at random, but ensure
    that resulting areas won't be too small.

    3) Create nodes for areas and bonds on the RHS. Create relations
    representing connections between them as specified by the RHS.

    4) Embed newly created sub-areas, that is, connect each new area
    to areas which were geometrically adjacent to the original matched
    area and are adjacent to this sub-area.

    5) Remove five nodes which represent the matched area and its bonds,
    and remove relations incident to these five nodes.

    Step (4) is implemented in a simplified way. Since the grammar is
    very simple, it is obvious that, for example, a recreation area
    will never have any northern, southern or western neighbours. So,
    in rules which divide a recreation area, embedding is done only
    on the eastern side.
    """

    def rule_p1(self):
        """
        Split S into two rooms sandwiched between two areas.
        """

        # Minimum and maximum rectangle sizes, etc:
        rec_xsize_min = 3
        hall_xsize = 2
        slp_xsize_min = 3
        kit_ysize_min = 2
        kit_ysize_max = 4
        xsize_min = rec_xsize_min + hall_xsize + slp_xsize_min
        ysize_min = 1 + kit_ysize_min

        # Find the starting area/rectangle:
        cond = 'b1.x2 - b1.x1 >= {0} AND b2.y2 - b2.y1 >= {1}'.format(xsize_min, ysize_min)
        rs = self.query_match_rect('S', cond)
        if len(rs) == 0:
            return None
        (s, llx, lly, urx, ury) = choice(rs)

        # Calculate output rectangle sizes in the X direction:
        xsize = urx - llx
        extra = xsize - xsize_min
        rec_xsize = rec_xsize_min + randint(0, extra)
        slp_xsize = xsize - rec_xsize - hall_xsize
        # ... and in the Y direction:
        ysize = ury - lly
        extra = ysize - 1 - kit_ysize_min
        kit_ysize = kit_ysize_min + randint(0, min(extra, kit_ysize_max - kit_ysize_min))
        hal_ysize = ysize - kit_ysize

        # Create replacement subrectangles:
        ra = self.query_create_rect('Recreation area', llx, lly, llx + rec_xsize, ury)
        h = self.query_create_rect('hall', llx + rec_xsize, lly, urx - slp_xsize, lly + hal_ysize)
        k = self.query_create_rect('kitchen', llx + rec_xsize, ury - kit_ysize, urx - slp_xsize, ury)
        sa = self.query_create_rect('Sleeping area', urx - slp_xsize, lly, urx, ury)
        self.query_create_edge('adj', ra.b2_id, h.b4_id)
        self.query_create_edge('adj', ra.b2_id, k.b4_id)
        self.query_create_edge('acc', h.b1_id, k.b3_id)
        self.query_create_edge('adj', sa.b4_id, h.b2_id)
        self.query_create_edge('adj', sa.b4_id, k.b2_id)

        # No embedding required because S has no neighbours.

        # Remove the original rectangle:
        self.query_delete_rect(s.id)

        return 'sub-areas created, cuts at X={0}, X={1}, Y={2}'.format(
                    llx + rec_xsize, urx - slp_xsize, lly + hal_ysize)

    def rule_p2(self):
        """
        Split a sleeping area (sa) into a bedroom (bed1), a bathroom
        (bath) and another bedroom (bed2).
        """

        bed_ysize_min = 2
        bath_ysize_min = 1
        bath_ysize_max = 2
        ysize_min = 2 * bed_ysize_min + bath_ysize_min

        rs = self.query_match_rect('Sleeping area', 'b2.y2 - b2.y1 >= {0}'.format(ysize_min))
        if len(rs) == 0:
            return None
        (sa, llx, lly, urx, ury) = choice(rs)

        extra = (ury - lly) - 2 * bed_ysize_min - bath_ysize_min
        bath_ysize = bath_ysize_min + randint(0, min(extra, bath_ysize_max - bath_ysize_min))
        extra = (ury - lly) - 2 * bed_ysize_min - bath_ysize
        bed1_ysize = bed_ysize_min + randint(0, extra)

        bed1 = self.query_create_rect('bedroom', llx, lly, urx, lly + bed1_ysize)
        bath = self.query_create_rect('bathroom', llx, lly + bed1_ysize, urx, lly + bed1_ysize + bath_ysize)
        bed2 = self.query_create_rect('bedroom', llx, lly + bed1_ysize + bath_ysize, urx, ury)
        self.query_create_edge('acc', bed1.b1_id, bath.b3_id)
        self.query_create_edge('adj', bath.b1_id, bed2.b3_id)

        self.query_embed_western(sa.id, bed1.id)
        self.query_embed_western(sa.id, bath.id)
        self.query_embed_western(sa.id, bed2.id)

        self.query_delete_rect(sa.id)

        return 'sub-areas created, cuts at Y={0}, Y={1}'.format(
                    lly + bed1_ysize, lly + bed1_ysize + bath_ysize)

    def rule_p3(self):
        """
        Split a sleeping area (sa) into a bathroom (bath) and a bedroom
        (bed).
        """

        bed_ysize_min = 2
        bath_ysize_min = 1
        bath_ysize_max = 2
        ysize_min = bed_ysize_min + bath_ysize_min

        rs = self.query_match_rect('Sleeping area', 'b2.y2 - b2.y1 >= {0}'.format(ysize_min))
        if len(rs) == 0:
            return None
        (sa, llx, lly, urx, ury) = choice(rs)

        extra = (ury - lly) - bed_ysize_min - bath_ysize_min
        bath_ysize = bath_ysize_min + randint(0, min(extra, bath_ysize_max - bath_ysize_min))

        bath = self.query_create_rect('bathroom', llx, lly, urx, lly + bath_ysize)
        bed = self.query_create_rect('bedroom', llx, lly + bath_ysize, urx, ury)
        self.query_create_edge('adj', bath.b1_id, bed.b3_id)

        self.query_embed_western(sa.id, bath.id)
        self.query_embed_western(sa.id, bed.id)

        self.query_delete_rect(sa.id)

        return 'sub-areas created, cut at Y={0}'.format(lly + bath_ysize)

    def rule_p4(self):
        """
        Split a recreation area (ra) into a living room (liv) and
        a terrace (ter).
        """

        liv_ysize_min = 2
        ter_ysize_min = 1
        ter_ysize_max = 3
        ysize_min = liv_ysize_min + ter_ysize_min

        rs = self.query_match_rect('Recreation area', 'b2.y2 - b2.y1 >= {}'.format(ysize_min))
        if len(rs) == 0:
            return None
        (ra, llx, lly, urx, ury) = choice(rs)

        extra = (ury - lly) - liv_ysize_min - ter_ysize_min
        ter_ysize = ter_ysize_min + randint(0, min(extra, ter_ysize_max - ter_ysize_min))

        liv = self.query_create_rect('living room', llx, lly, urx, ury - ter_ysize)
        ter = self.query_create_rect('terrace', llx, ury - ter_ysize, urx, ury)
        self.query_create_edge('acc', liv.b1_id, ter.b3_id)

        self.query_embed_eastern(ra.id, liv.id)
        self.query_embed_eastern(ra.id, ter.id)

        self.query_delete_rect(ra.id)

        return 'sub-areas created, cut at Y={}'.format(ury - ter_ysize)

    def rule_p5(self):
        """
        Split a recreation area (ra) into a living room (liv), a dining
        room (din) and a terrace (ter).
        """

        liv_ysize_min = 2
        din_ysize_min = 2
        ter_ysize_min = 1
        ter_ysize_max = 3
        ysize_min = liv_ysize_min + din_ysize_min + ter_ysize_min

        rs = self.query_match_rect('Recreation area', 'b2.y2 - b2.y1 >= {}'.format(ysize_min))
        if len(rs) == 0:
            return None
        (ra, llx, lly, urx, ury) = choice(rs)

        extra = (ury - lly) - liv_ysize_min - din_ysize_min - ter_ysize_min
        ter_ysize = ter_ysize_min + randint(0, min(extra, ter_ysize_max - ter_ysize_min))
        extra = (ury - lly) - liv_ysize_min - din_ysize_min - ter_ysize
        liv_ysize = liv_ysize_min + randint(0, extra)

        liv = self.query_create_rect('living room', llx, lly, urx, lly + liv_ysize)
        din = self.query_create_rect('dining room', llx, lly + liv_ysize, urx, ury - ter_ysize)
        ter = self.query_create_rect('terrace', llx, ury - ter_ysize, urx, ury)
        self.query_create_edge('acc', liv.b1_id, din.b3_id)
        self.query_create_edge('acc', din.b1_id, ter.b3_id)

        self.query_embed_eastern(ra.id, liv.id)
        self.query_embed_eastern(ra.id, din.id)
        self.query_embed_eastern(ra.id, ter.id)

        self.query_delete_rect(ra.id)

        return 'sub-areas created, cuts at Y={0}, Y={1}'.format(
                    lly + liv_ysize, ury - ter_ysize)

    def rule_p6(self):
        """
        Split a recreation area (ra) into a living room (liv) and
        a dining room (din).
        """

        # Minimum acceptable rectangle dimensions in the Y direction:
        liv_ysize_min = 2
        din_ysize_min = 2
        ysize_min = liv_ysize_min + din_ysize_min

        # Find a rectangle representing a recreation area:
        rs = self.query_match_rect('Recreation area', 'b2.y2 - b2.y1 >= {}'.format(ysize_min))
        if len(rs) == 0:
            return None
        (ra, llx, lly, urx, ury) = choice(rs)

        # Pick the living room's Y-dimension:
        extra = (ury - lly) - liv_ysize_min - din_ysize_min
        liv_ysize = liv_ysize_min + randint(0, extra)

        # Create replacement rectangles and edges:
        liv = self.query_create_rect('living room', llx, lly, urx, lly + liv_ysize)
        din = self.query_create_rect('dining room', llx, lly + liv_ysize, urx, ury)
        self.query_create_edge('acc', liv.b1_id, din.b3_id)

        # Embed newly created rectangles:
        self.query_embed_eastern(ra.id, liv.id)
        self.query_embed_eastern(ra.id, din.id)

        # Remove the original rectangle and its incident edges:
        self.query_delete_rect(ra.id)

        return 'sub-areas created, cut at Y={0}'.format(lly + liv_ysize)


def main():
    driver = neo4j.Driver('bolt://localhost:7687', auth=('neo4j', 'neo4j'))
    g = ExampleGrammar(driver, draw_grid=True)
    # g = ExampleGrammar(driver, room_padding=1, draw_edges=True)
    with open('deriv.log', 'w') as f:
        g.generate_and_multipage_save('deriv', ppi=288, flog=f)
        # f.write('\n')
        # g.dump_graph(flog=f)
    g.save_pdf('final.pdf')
    g.save_png('final.png', ppi=288)
    driver.close()

if __name__ == '__main__':
    main()
