/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* This is a native C implementation.  See this:
    http://www.dgp.toronto.edu/~ah/csc418/fall_2001/notes/scanconv.html
    http://www.cc.gatech.edu/gvu/multimedia/nsfmmedia/graphics/elabor/polyscan/polyscan1.html
    http://graphics.cs.ucdavis.edu/education/GraphicsNotes/Scan-Conversion/Scan-Conversion.html
    http://www.cs.berkeley.edu/~sequin/CS184/TEXT/Algorithm.html
  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct edge Edge;
struct edge {
    int ymax;
    float x;
    float xincr;
    Edge* next;
};

void
remove_old_edges (Edge** head, int y)
{
    Edge *p, *n;
    p = *head;
    while (p && p->ymax < y) p = p->next;
    *head = p;
    while (p) {
	n = p->next;
	while (n && n->ymax < y) n = n->next;
	p->next = n;
	p = n;
    }
}

void
insert_ordered_by_x (Edge** head, Edge* c)
{
    Edge* p;
    if ((p = *head)) {
	if (p->x > c->x) {
	    c->next = p;
	    *head = c;
	} else {
	    while (p->next && p->next->x < c->x) p = p->next;
	    c->next = p->next;
	    p->next = c;
	}
    } else {
	*head = c;
	c->next = 0;
    }
}

void
print_edges (Edge* p)
{
    while (p) {
	printf ("[%g %g %d] ", p->x, p->xincr, p->ymax);
	p = p->next;
    }
}


void
render_slice_polyline (unsigned char* acc_img,
		    int* dims,
		    float* spacing,
		    float* offset,
		    int num_vertices,
		    float* x,
		    float* y)
{
    unsigned char* imgp;
    Edge** edge_table;
    Edge* edge_list;	    /* Global edge list */
    Edge* ael;		    /* Active edge list */
    int i;

    /* Check if last vertex == first vertex.  If so, remove it. */
    if (x[num_vertices-1]==x[0] && y[num_vertices-1]==y[0]) {
	num_vertices --;
    }

    /* Destructively convert to image coordinates */
    for (i = 0; i < num_vertices; i++) {
	x[i] = (x[i] - offset[0]) / spacing[0];
	y[i] = (y[i] - offset[1]) / spacing[1];
    }

    /* Make edge table */
    edge_table = (Edge**) malloc (dims[1] * sizeof(Edge*));
    edge_list = (Edge*) malloc (num_vertices * sizeof(Edge));
    memset (edge_table, 0, dims[1] * sizeof(Edge*));
    for (i = 0; i < num_vertices; i++) {
	int ymin, ymax;
	int a = i, b = (i==num_vertices-1 ? 0 : i+1);
	/* Reorder segment so that y[a] > y[b] */
	if (y[a] == y[b]) continue;
	if (y[a] < y[b]) a = b, b = i;
	/* Reject segments too high or too low */
	ymin = (int) ceil(y[b]);
	if (ymin > dims[1]-1) continue;
	ymax = (int) floor(y[a]);
	if (ymax < 0) continue;
	/* If upper y lies on scan line, don't count it as an intersection */
	if (y[a] == ymax) ymax --;
	/* Reject segments that don't intersect a scan line */
	if (ymax < ymin) continue;
	/* Clip segments against image boundary */
	if (ymin < 0) ymin = 0;
	if (ymax > dims[1]-1) ymax = dims[1]-1;
	/* Shorten the segment & fill in edge data */
	edge_list[i].ymax = ymax;
	edge_list[i].xincr = (x[a] - x[b]) / (y[a] - y[b]);
	edge_list[i].x = x[b] + (ymin - y[b]) * edge_list[i].xincr;
	edge_list[i].next = 0;
	/* Insert into edge_table */
#if defined (commentout)
	printf ("[y:%g %g, x:%g %g] -> [y:%d %d, x:%g (%g)]\n", y[b], y[a], x[b], x[a],
		ymin, ymax, edge_list[i].x, edge_list[i].xincr);
#endif
        insert_ordered_by_x (&edge_table[ymin], &edge_list[i]);
    }

    /* Debug edge table */
#if defined (commentout)
    printf ("-------------------------------------------\n");
    for (i = 0; i < dims[1]; i++) {
	if (edge_table[i]) {
	    printf ("%d: ", i);
	    print_edges (edge_table[i]);
	    printf ("\n");
	}
    }
#endif

    /* Loop through scanline, rendering each */
    imgp = acc_img;
    ael = 0;
    for (i = 0; i < dims[1]; i++) {
	int x, num_crossings;
	Edge *n, *c;
	/* Remove old edges from AEL */
	remove_old_edges (&ael, i);

	/* Add new edges to AEL */
	c = edge_table[i];
	while (c) {
	    n = c->next;
	    insert_ordered_by_x (&ael, c);
	    c = n;
	}

	/* Count scan intersections & rasterize */
	num_crossings = 0;
	x = 0;
	c = ael;
#if defined (commentout)
	printf ("%d ", i);
	print_edges (ael);
#endif
	while (x < dims[0]) {
	    int next_x;
	    while (1) {
		if (!c) {
		    next_x = dims[0];
		    break;
		} else if (x >= c->x) {
		    c = c->next;
		    num_crossings ++;
		    continue;
		} else {
		    next_x = (int) floor (c->x) + 1;
		    if (next_x > dims[0]) next_x = dims[0];
		    break;
		}
	    }
	    num_crossings = num_crossings % 2;
#if defined (commentout)
	    printf ("(%d %c %d)", x, num_crossings?'+':'-', next_x-1);
#endif
	    while (x < next_x) {
		*imgp++ = num_crossings;
		x++;
	    }
	}
#if defined (commentout)
	printf ("\n");
	getchar();
#endif

	/* Update x values on AEL */
	c = ael;
	while (c) {
	    c->x += c->xincr;
	    c = c->next;
	}

	/* Resort AEL - this could be done more efficiently */
	c = ael;
	while (c) {
	    if (c->next && c->x > c->next->x) {
		Edge* tmp = c->next;
		c->next = c->next->next;
		insert_ordered_by_x (&ael, tmp);
		c = tmp;
	    } else {
		c = c->next;
	    }
	}
    }

    /* Free things up */
    free (edge_table);
    free (edge_list);
}
